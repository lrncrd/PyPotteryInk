"""
Project Manager for PyPotteryInk.

A project is a folder `projects/<project_id>/` that holds everything of one job:

    project.json            metadata, the drawing library and the history of every run
    drawings/original/      the drawings imported into the project (the only input of every tab)
    drawings/preprocessed/  the brightness/contrast-corrected version of a drawing, when made
    thumbnails/             small previews of the library
    outputs/<mode>/<label>/ inked results, one folder per combination of settings
    diagnostics/ stats/ references/

So the user sees, on a later session, which pieces are already done and how. Same
on-disk contract as the Scan/Lens/Trace managers (`projects/<id>/project.json`), so
the launcher backup picks it up.
"""

import hashlib
import json
import re
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
DRAWING_VERSIONS = ('original', 'preprocessed')
THUMB_WIDTH = 240
PROJECT_FOLDERS = ('drawings/original', 'drawings/preprocessed', 'thumbnails', 'outputs/local',
                   'outputs/cloud', 'references', 'diagnostics', 'stats')


def _safe_filename(filename: str) -> str:
    """Keep the base name only, with letters, digits, '.', '-', '_' (like werkzeug.secure_filename)."""
    base = re.split(r'[\\/]', str(filename))[-1]
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', base.replace(' ', '_')).strip('._')
    return base[:150]


def _slug(text, limit: int = 40) -> str:
    return re.sub(r'[^A-Za-z0-9.]+', '-', str(text)).strip('-.')[:limit] or 'x'


def _num(value) -> str:
    """1.0 -> '1', 0.75 -> '0.75': compact and stable in folder names."""
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    except (TypeError, ValueError):
        return _slug(value, 10)


def config_label(mode: str, params: Dict) -> str:
    """Folder name for one combination of parameters, e.g. `6h-MCG_p512_o64_c1_x1` or
    `gpt-image-2_fine_v1_ref-sheet-a1b2c3`. Same parameters always give the same name, so a
    rerun with the same settings replaces its results and any other settings keep theirs."""
    if mode == 'local':
        parts = [_slug(str(params.get('model', 'model')).replace('.pkl', '')),
                 f"p{_num(params.get('patch_size'))}", f"o{_num(params.get('overlap'))}",
                 f"c{_num(params.get('contrast_scale'))}", f"x{_num(params.get('upscale'))}"]
        if params.get('use_fp16'):
            parts.append('fp16')
        if params.get('input') == 'preprocessed':
            parts.append('pre')
        return '_'.join(parts)
    if mode == 'cloud':
        parts = [_slug(str(params.get('model', 'model')).split('/')[-1]), _slug(params.get('style', 'fine'), 12),
                 f"v{_num(params.get('variants', 1))}"]
        reference = params.get('reference') or 'none'
        if reference != 'none':
            ref_hash = re.sub(r'[^0-9a-f]', '', str(params.get('reference_hash') or '').lower())[:6]
            parts.append(f"ref-{_slug(reference, 8)}" + (f"-{ref_hash}" if ref_hash else ''))
        extra = str(params.get('extra_prompt') or '').strip()
        if extra:
            parts.append('extra-' + hashlib.sha1(extra.encode('utf-8')).hexdigest()[:6])
        if params.get('align'):   # off by default: only a registered run gets the suffix
            parts.append('aligned')
        if params.get('input') == 'preprocessed':
            parts.append('pre')
        return '_'.join(parts)
    raise ValueError(f"Unknown mode: {mode}")


class ProjectManager:
    """Creates, lists and deletes projects and keeps their run history."""

    def __init__(self, projects_root: str = "projects"):
        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(exist_ok=True)
        self._lock = threading.Lock()  # runs are recorded from background threads

    # ------------------------------------------------------------------ paths
    def get_project_path(self, project_id: str) -> Optional[Path]:
        """The project folder, or None when the id is not a project of this root."""
        if not project_id or project_id in ('.', '..') or Path(project_id).name != project_id:
            return None  # blocks '..', separators and absolute paths from the URL
        path = self.projects_root / project_id
        if path.resolve().parent != self.projects_root.resolve():
            return None  # e.g. a symlink pointing out of the projects root
        return path if (path / 'project.json').exists() else None

    def outputs_dir(self, project_id: str, mode: str, label: str = "") -> Optional[Path]:
        """Where a run saves inside the project: `outputs/<mode>/<label>`, one folder per
        combination of parameters (see `config_label`). Without a label, the mode folder."""
        path = self.get_project_path(project_id)
        if path is None or mode not in ('local', 'cloud'):
            return None
        base = path / 'outputs' / mode
        return (base / label).resolve() if label else base.resolve()

    # --------------------------------------------------------------- projects
    def create_project(self, project_name: str, description: str = "") -> Dict:
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_') or 'project'

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{safe_name}_{timestamp}"
        project_path = self.projects_root / project_id
        if project_path.exists():
            raise ValueError(f"Project already exists: {project_id}")

        for folder in PROJECT_FOLDERS:
            (project_path / folder).mkdir(parents=True, exist_ok=True)

        now = datetime.now().isoformat()
        metadata = {
            'project_id': project_id,
            'project_name': project_name,
            'description': description,
            'created_at': now,
            'last_modified': now,
            'settings': {'external_dir': ''},   # optional second copy of every run's outputs
            'drawings': {},                       # name -> {sha1, size, imported_at, preprocessed}
            'runs': [],
        }
        self._save_metadata(project_path, metadata)
        return metadata

    def list_projects(self) -> List[Dict]:
        projects = []
        self.projects_root.mkdir(exist_ok=True)
        for project_dir in self.projects_root.iterdir():
            if project_dir.is_dir() and (project_dir / 'project.json').exists():
                try:
                    projects.append(self._summary(self._load_metadata(project_dir)))
                except Exception as e:
                    print(f"Error loading project {project_dir.name}: {e}")
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        return projects

    def get_project(self, project_id: str) -> Optional[Dict]:
        path = self.get_project_path(project_id)
        if path is None:
            return None
        try:
            metadata = self._load_metadata(path)
            metadata.setdefault('settings', {}).setdefault('external_dir', '')
            metadata.setdefault('drawings', {})
            return metadata
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None

    def update_settings(self, project_id: str, updates: Dict) -> Optional[Dict]:
        """Merge project settings (today: `external_dir`). Returns the settings, or None if unknown."""
        with self._lock:
            path = self.get_project_path(project_id)
            if path is None:
                return None
            metadata = self._load_metadata(path)
            settings = metadata.setdefault('settings', {})
            settings.update(updates)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(path, metadata)
            return settings

    def delete_project(self, project_id: str) -> bool:
        path = self.get_project_path(project_id)
        if path is None:
            return False
        try:
            shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Error deleting project {project_id}: {e}")
            return False

    # ---------------------------------------------------------------- library
    def import_drawing(self, project_id: str, filename: str, stream: BinaryIO) -> Optional[Dict]:
        """Copy one drawing into the library. Returns {status: added|duplicate|renamed|skipped, name}.
        The same content already in the library is not copied twice; the same name with other
        content gets a numbered name, so nothing in the project is ever replaced by an import."""
        path = self.get_project_path(project_id)
        if path is None:
            return None
        name = _safe_filename(filename)
        if not name or not name.lower().endswith(IMAGE_EXTENSIONS):
            return {'status': 'skipped', 'name': str(filename), 'reason': 'not a supported image'}
        folder = path / 'drawings' / 'original'
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f'.import-{uuid.uuid4().hex}'
        digest = hashlib.sha1()
        size = 0
        with open(tmp, 'wb') as out:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                size += len(block)
                out.write(block)
        sha1 = digest.hexdigest()
        with self._lock:
            metadata = self._load_metadata(path)
            drawings = metadata.setdefault('drawings', {})
            for existing, info in drawings.items():
                if info.get('sha1') == sha1:
                    tmp.unlink()
                    return {'status': 'duplicate', 'name': existing, 'requested': name}
            final, status = name, 'added'
            stem, ext = Path(name).stem, Path(name).suffix
            n = 2
            while final in drawings or (folder / final).exists():
                final, status = f"{stem}_{n}{ext}", 'renamed'
                n += 1
            tmp.replace(folder / final)
            drawings[final] = {'sha1': sha1, 'size': size,
                               'imported_at': datetime.now().isoformat(timespec='seconds'),
                               'preprocessed': None}
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(path, metadata)
        return {'status': status, 'name': final, 'requested': name}

    def list_drawings(self, project_id: str) -> Optional[List[Dict]]:
        """The library, sorted by name, each with the settings (configs) that already inked it."""
        metadata = self.get_project(project_id)
        if metadata is None:
            return None
        index = self.processed_index(project_id)
        out = []
        for name in sorted(metadata['drawings'], key=str.lower):
            info = metadata['drawings'][name]
            out.append({'name': name, 'sha1': info.get('sha1'), 'size': info.get('size'),
                        'imported_at': info.get('imported_at'),
                        'preprocessed': bool(info.get('preprocessed')),
                        'preprocessed_at': (info.get('preprocessed') or {}).get('at'),
                        'inked': sorted(index.get(info.get('sha1'), {}))})
        return out

    def drawing_path(self, project_id: str, name: str, version: str = 'original') -> Optional[Path]:
        """File of a library drawing, or None when the name is not in the library.
        Only names recorded in project.json are served, so a URL can never reach other files."""
        path = self.get_project_path(project_id)
        if path is None or version not in DRAWING_VERSIONS:
            return None
        info = self._load_metadata(path).get('drawings', {}).get(name)
        if info is None or (version == 'preprocessed' and not info.get('preprocessed')):
            return None
        file = path / 'drawings' / version / name
        return file if file.is_file() else None

    def input_path(self, project_id: str, name: str, use_preprocessed: bool):
        """(file, version) a run reads: the preprocessed version when asked for and available."""
        if use_preprocessed:
            file = self.drawing_path(project_id, name, 'preprocessed')
            if file is not None:
                return file, 'preprocessed'
        return self.drawing_path(project_id, name, 'original'), 'original'

    def set_preprocessed(self, project_id: str, name: str, info: Optional[Dict]) -> bool:
        """Record (or clear, with None) the preprocessed version of a drawing."""
        with self._lock:
            path = self.get_project_path(project_id)
            if path is None:
                return False
            metadata = self._load_metadata(path)
            drawing = metadata.get('drawings', {}).get(name)
            if drawing is None:
                return False
            drawing['preprocessed'] = info
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(path, metadata)
        self._thumb_file(path, name, 'preprocessed').unlink(missing_ok=True)
        return True

    def remove_drawing(self, project_id: str, name: str) -> bool:
        """Take a drawing out of the library. Its inked results and the history stay in the project."""
        with self._lock:
            path = self.get_project_path(project_id)
            if path is None:
                return False
            metadata = self._load_metadata(path)
            if name not in metadata.get('drawings', {}):
                return False
            del metadata['drawings'][name]
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(path, metadata)
        for version in DRAWING_VERSIONS:
            (path / 'drawings' / version / name).unlink(missing_ok=True)
            self._thumb_file(path, name, version).unlink(missing_ok=True)
        return True

    def thumbnail(self, project_id: str, name: str, version: str = 'original') -> Optional[Path]:
        """A small PNG preview of a library drawing, made once and cached (TIFF included)."""
        source = self.drawing_path(project_id, name, version)
        if source is None:
            return None
        thumb = self._thumb_file(self.get_project_path(project_id), name, version)
        if not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime:
            from PIL import Image, ImageOps
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as img:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 2))
                img.convert('L').save(thumb, 'PNG')
        return thumb

    @staticmethod
    def _thumb_file(project_path: Path, name: str, version: str) -> Path:
        return project_path / 'thumbnails' / version / (name + '.png')

    def area_dir(self, project_id: str, area: str) -> Optional[Path]:
        """Project folders the browser may read files from (galleries, previews)."""
        path = self.get_project_path(project_id)
        folders = {'outputs': 'outputs', 'diagnostics': 'diagnostics',
                   'original': 'drawings/original', 'preprocessed': 'drawings/preprocessed'}
        if path is None or area not in folders:
            return None
        return (path / folders[area]).resolve()

    def ensure_folders(self, project_id: str) -> None:
        """Projects made by an earlier version lack some folders."""
        path = self.get_project_path(project_id)
        if path is not None:
            for folder in PROJECT_FOLDERS:
                (path / folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- runs
    def add_run(self, project_id: str, mode: str, params: Dict, items: List[Dict],
                output_dir: str = "", extra: Optional[Dict] = None) -> Optional[Dict]:
        """Append a finished run. `items` are the drawings it touched:
        {source, sha1, outputs: [files], status: 'ok'|'failed', seconds, cost_usd}.
        `config` ("<mode>/<label>", the folder under outputs/) says which settings produced them."""
        with self._lock:
            path = self.get_project_path(project_id)
            if path is None:
                return None
            metadata = self._load_metadata(path)
            run = {
                'run_id': uuid.uuid4().hex[:8],
                'mode': mode,
                'config': f"{mode}/{config_label(mode, params)}",
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'params': params,
                'output_dir': output_dir,
                'items': items,
            }
            if extra:
                run.update(extra)
            metadata.setdefault('runs', []).append(run)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(path, metadata)
            return run

    def processed_index(self, project_id: str) -> Dict[str, Dict[str, Dict]]:
        """sha1 -> { config -> latest successful run of that drawing with those settings }.
        A drawing done with another model is not "done" for this one, so the UI compares configs.
        Each record keeps the parameters, so the UI can show how it was processed."""
        metadata = self.get_project(project_id)
        index: Dict[str, Dict[str, Dict]] = {}
        if not metadata:
            return index
        for run in metadata.get('runs', []):  # chronological: later runs overwrite earlier ones
            run_config = run.get('config') or f"{run.get('mode')}/{config_label(run.get('mode'), run.get('params', {}))}"
            for item in run.get('items', []):
                if item.get('status') != 'ok' or not item.get('sha1'):
                    continue
                # each drawing records its own folder: a run can mix original and preprocessed inputs
                config = item.get('config') or run_config
                index.setdefault(item['sha1'], {})[config] = {
                    'source': item.get('source'),
                    'version': item.get('version', 'original'),
                    'run_id': run.get('run_id'),
                    'mode': run.get('mode'),
                    'config': config,
                    'timestamp': run.get('timestamp'),
                    'params': run.get('params', {}),
                    'outputs': item.get('outputs', []),
                }
        return index

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _summary(metadata: Dict) -> Dict:
        """Card data for the project list: no heavy run history."""
        runs = metadata.get('runs', [])
        done = set()
        for run in runs:
            done.update(i.get('sha1') for i in run.get('items', []) if i.get('status') == 'ok' and i.get('sha1'))
        summary = {k: metadata.get(k) for k in
                   ('project_id', 'project_name', 'description', 'created_at', 'last_modified')}
        summary['drawings_count'] = len(metadata.get('drawings', {}))
        summary['runs_count'] = len(runs)
        summary['pieces_done'] = len(done)
        summary['last_run'] = runs[-1].get('timestamp') if runs else None
        return summary

    @staticmethod
    def _load_metadata(project_path: Path) -> Dict:
        with open(project_path / 'project.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _save_metadata(project_path: Path, metadata: Dict) -> None:
        tmp = project_path / 'project.json.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        tmp.replace(project_path / 'project.json')  # a crash mid-write cannot corrupt the history
