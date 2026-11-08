import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_manager import ProjectManager


def make(tmp_path):
    return ProjectManager(projects_root=str(tmp_path / "projects"))


def test_create_list_delete(tmp_path):
    pm = make(tmp_path)
    meta = pm.create_project("Tell el-Hammam / 2026", "campaign")
    pid = meta["project_id"]

    assert "/" not in pid and " " not in pid
    assert (tmp_path / "projects" / pid / "project.json").exists()
    assert pm.outputs_dir(pid, "local").is_dir()
    assert pm.outputs_dir(pid, "cloud").is_dir()
    assert [p["project_id"] for p in pm.list_projects()] == [pid]

    assert pm.delete_project(pid) is True
    assert pm.list_projects() == []
    assert pm.delete_project(pid) is False


def test_ids_cannot_escape_the_projects_root(tmp_path):
    pm = make(tmp_path)
    (tmp_path / "project.json").write_text("{}")  # a decoy one level above the root
    for bad in ("..", "../", "", "a/b", str(tmp_path)):
        assert pm.get_project_path(bad) is None
        assert pm.get_project(bad) is None
        assert pm.delete_project(bad) is False


def test_add_run_and_processed_index(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("p")["project_id"]

    local = {"model": "model", "patch_size": 512, "overlap": 64, "contrast_scale": 1.0, "upscale": 1}
    pm.add_run(pid, "local", local, [
        {"source": "a.png", "sha1": "aaa", "outputs": ["a.png"], "status": "ok"},
        {"source": "b.png", "sha1": "bbb", "outputs": [], "status": "failed"},
    ])
    pm.add_run(pid, "cloud", {"model": "openai/gpt-image-2", "style": "fine", "variants": 1}, [
        {"source": "a.png", "sha1": "aaa", "outputs": ["a_v1.png"], "status": "ok", "cost_usd": 0.05},
    ])

    index = pm.processed_index(pid)
    assert set(index) == {"aaa"}                      # failed pieces are not "done"
    # the same drawing is kept once per combination of settings
    assert set(index["aaa"]) == {"local/model_p512_o64_c1_x1", "cloud/gpt-image-2_fine_v1"}
    assert index["aaa"]["cloud/gpt-image-2_fine_v1"]["params"]["model"] == "openai/gpt-image-2"

    summary = pm.list_projects()[0]
    assert summary["runs_count"] == 2 and summary["pieces_done"] == 1
    assert "runs" not in summary


def test_add_run_on_unknown_project_is_a_noop(tmp_path):
    assert make(tmp_path).add_run("nope", "local", {}, []) is None


def test_project_json_is_valid_utf8_and_has_no_temp_leftover(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("Ceramica è bella")["project_id"]
    pm.add_run(pid, "cloud", {"extra_prompt": "linee più nette"}, [])
    folder = tmp_path / "projects" / pid
    data = json.loads((folder / "project.json").read_text(encoding="utf-8"))
    assert data["runs"][0]["params"]["extra_prompt"] == "linee più nette"
    assert not (folder / "project.json.tmp").exists()


def test_config_label_local():
    base = {"model": "6h-MCG Model", "patch_size": 512, "overlap": 64, "contrast_scale": 1.0, "upscale": 1.0}
    from project_manager import config_label
    assert config_label("local", base) == "6h-MCG-Model_p512_o64_c1_x1"
    assert config_label("local", {**base, "use_fp16": True}).endswith("_fp16")
    assert config_label("local", {**base, "contrast_scale": 1.5}) != config_label("local", base)
    assert config_label("local", {**base, "model": "other.pkl"}) != config_label("local", base)


def test_config_label_cloud_separates_every_setting_that_changes_the_result():
    from project_manager import config_label
    base = {"model": "openai/gpt-image-2", "style": "fine", "variants": 1, "reference": "none", "align": False}
    plain = config_label("cloud", base)
    assert plain == "gpt-image-2_fine_v1"
    labels = {
        plain,
        config_label("cloud", {**base, "style": "coarse"}),
        config_label("cloud", {**base, "variants": 2}),
        config_label("cloud", {**base, "extra_prompt": "thin lines"}),
        config_label("cloud", {**base, "extra_prompt": "thick lines"}),
        config_label("cloud", {**base, "reference": "sheet", "reference_hash": "aabbccdd"}),
        config_label("cloud", {**base, "reference": "sheet", "reference_hash": "11223344"}),
        config_label("cloud", {**base, "reference": "style", "reference_hash": "aabbccdd"}),
        config_label("cloud", {**base, "align": True}),
    }
    assert len(labels) == 9
    # budget and surrounding whitespace do not change the result, so they do not open a new folder
    assert config_label("cloud", {**base, "budget_usd": 5, "extra_prompt": "   "}) == plain


def test_config_label_is_a_safe_folder_name():
    from project_manager import config_label
    label = config_label("local", {"model": r"../../evil\name:*?.pkl", "patch_size": 512, "overlap": 64,
                                   "contrast_scale": 1, "upscale": 1})
    assert not any(c in label for c in r'/\:*?"<>|') and ".." not in label


def test_outputs_dir_per_config_and_settings(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("p")["project_id"]
    folder = pm.outputs_dir(pid, "cloud", "gpt-image-2_fine_v1")
    assert folder == (tmp_path / "projects" / pid / "outputs" / "cloud" / "gpt-image-2_fine_v1").resolve()
    assert pm.outputs_dir(pid, "other", "x") is None
    assert pm.get_project(pid)["settings"]["external_dir"] == ""
    assert pm.update_settings(pid, {"external_dir": "D:/shared"})["external_dir"] == "D:/shared"
    assert pm.get_project(pid)["settings"]["external_dir"] == "D:/shared"
    assert pm.update_settings("nope", {"external_dir": "x"}) is None


def _png(shade=128, size=(80, 60)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", size, shade).save(buf, "PNG")
    buf.seek(0)
    return buf


def test_import_dedup_rename_and_skip(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("lib")["project_id"]
    assert pm.import_drawing(pid, "C:/scans/Piece 1.png", _png(10))["status"] == "added"
    dup = pm.import_drawing(pid, "copy-of-piece.png", _png(10))
    assert dup == {"status": "duplicate", "name": "Piece_1.png", "requested": "copy-of-piece.png"}
    renamed = pm.import_drawing(pid, "Piece 1.png", _png(200))
    assert renamed["status"] == "renamed" and renamed["name"] == "Piece_1_2.png"
    assert pm.import_drawing(pid, "notes.txt", _png())["status"] == "skipped"
    names = [d["name"] for d in pm.list_drawings(pid)]
    assert names == ["Piece_1.png", "Piece_1_2.png"]
    assert not [f for f in (tmp_path / "projects" / pid / "drawings" / "original").iterdir() if f.name.startswith(".import")]


def test_library_paths_only_serve_known_names(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("lib")["project_id"]
    pm.import_drawing(pid, "a.png", _png())
    assert pm.drawing_path(pid, "a.png").is_file()
    for bad in ("../project.json", r"..\project.json", "b.png", ""):
        assert pm.drawing_path(pid, bad) is None
    assert pm.drawing_path(pid, "a.png", "preprocessed") is None
    assert pm.drawing_path(pid, "a.png", "../x") is None
    assert pm.area_dir(pid, "outputs").is_dir() and pm.area_dir(pid, "secrets") is None


def test_preprocessed_version_and_input_path(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("lib")["project_id"]
    pm.import_drawing(pid, "a.png", _png())
    assert pm.input_path(pid, "a.png", True)[1] == "original"     # nothing preprocessed yet
    pre = tmp_path / "projects" / pid / "drawings" / "preprocessed" / "a.png"
    pre.write_bytes(_png(90).read())
    assert pm.set_preprocessed(pid, "a.png", {"adjustments": ["contrast"]})
    assert pm.input_path(pid, "a.png", True) == (pre, "preprocessed")
    assert pm.input_path(pid, "a.png", False)[1] == "original"
    assert pm.list_drawings(pid)[0]["preprocessed"] is True


def test_thumbnails_png_and_tiff(tmp_path):
    import io
    from PIL import Image
    pm = make(tmp_path)
    pid = pm.create_project("lib")["project_id"]
    pm.import_drawing(pid, "big.png", _png(size=(1200, 900)))
    tif = io.BytesIO(); Image.new("RGB", (600, 800), "white").save(tif, "TIFF"); tif.seek(0)
    pm.import_drawing(pid, "scan.tif", tif)
    for name in ("big.png", "scan.tif"):
        with Image.open(pm.thumbnail(pid, name)) as t:
            assert t.width <= 240 and t.format == "PNG"
    assert pm.thumbnail(pid, "missing.png") is None


def test_remove_drawing_keeps_history(tmp_path):
    pm = make(tmp_path)
    pid = pm.create_project("lib")["project_id"]
    sha = pm.import_drawing(pid, "a.png", _png()) and pm.list_drawings(pid)[0]["sha1"]
    pm.add_run(pid, "local", {"model": "m", "patch_size": 512, "overlap": 64, "contrast_scale": 1, "upscale": 1},
               [{"source": "a.png", "sha1": sha, "outputs": ["a.png"], "status": "ok"}])
    assert pm.list_drawings(pid)[0]["inked"] == ["local/m_p512_o64_c1_x1"]
    assert pm.remove_drawing(pid, "a.png") and pm.list_drawings(pid) == []
    assert len(pm.get_project(pid)["runs"]) == 1
    assert pm.remove_drawing(pid, "a.png") is False


def test_preprocessed_input_gets_its_own_folder():
    from project_manager import config_label
    local = {"model": "m", "patch_size": 512, "overlap": 64, "contrast_scale": 1, "upscale": 1}
    assert config_label("local", {**local, "input": "preprocessed"}) == config_label("local", local) + "_pre"
    cloud = {"model": "openai/gpt-image-2", "style": "fine", "variants": 1}
    assert config_label("cloud", {**cloud, "input": "preprocessed"}).endswith("_pre")
    assert config_label("cloud", {**cloud, "input": "original"}) == "gpt-image-2_fine_v1"
