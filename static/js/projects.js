// PyPotteryInk: projects. Every tab works inside an open project: the drawings come from the project
// library (imported once), every run is recorded with its settings, and each combination of settings
// saves into its own folder. This file owns the project shell (gating of the work tabs, the Projects
// tab, the library) and the library picker the four work tabs use to choose drawings. Loaded before
// app.js, which reads the selection through window.PyProjects.
(function () {
    const $ = id => document.getElementById(id);
    const STORAGE_KEY = 'pypotteryink.currentProject';
    const IMAGE_RE = /\.(png|jpe?g|tiff?)$/i;
    const UPLOAD_BATCH = 20;

    let current = null;          // { id, name, path, project, drawings: [...] }
    const status = { local: {}, cloud: {} };      // name -> { state, version, record } for the settings on screen
    const labels = { local: null, cloud: null };   // output folders of the settings on screen
    const statusTokens = { local: 0, cloud: 0 };
    const pickers = {};

    // ---- small helpers ------------------------------------------------------------------
    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function icon(name) {
        const i = document.createElement('i');
        i.className = `bi bi-${name}`;
        i.setAttribute('aria-hidden', 'true');
        return i;
    }

    async function api(url, options) {
        const response = await fetch(url, options);
        const data = await response.json().catch(() => ({}));
        if (!response.ok || data.success === false) throw new Error(data.error || `Request failed (${response.status})`);
        return data;
    }

    function jsonBody(method, payload) {
        return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
    }

    const projectUrl = path => `/api/projects/${encodeURIComponent(current.id)}${path}`;
    const encodePath = rel => String(rel).split('/').map(encodeURIComponent).join('/');

    function store(value) {
        try {
            if (value) localStorage.setItem(STORAGE_KEY, value); else localStorage.removeItem(STORAGE_KEY);
        } catch (e) { /* private window or blocked storage: the project is simply not remembered */ }
    }
    function stored() {
        try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
    }

    function fmtDate(iso) {
        return iso ? iso.replace('T', ' ').slice(0, 16) : '';
    }

    const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

    function describeParams(mode, p) {
        p = p || {};
        const input = p.use_preprocessed === false ? 'originals' : 'preprocessed when available';
        if (mode === 'cloud') {
            const parts = [p.model || '?', `${p.style || 'fine'} stippling`, plural(p.variants || 1, 'version'),
                p.reference && p.reference !== 'none' ? `reference: ${p.reference}` : 'no reference',
                p.align ? 'registered' : 'not registered', `input: ${input}`];
            if (p.budget_usd) parts.push(`stop at $${p.budget_usd}`);
            if (p.extra_prompt) parts.push(`extra: "${p.extra_prompt}"`);
            return parts.join(' · ');
        }
        return [p.model || '?', `contrast ${p.contrast_scale}`, `patch ${p.patch_size}`, `overlap ${p.overlap}`,
            `scale ${p.upscale}×`, p.use_fp16 ? 'fp16' : null, `input: ${input}`].filter(Boolean).join(' · ');
    }

    // ---- the settings now on screen ------------------------------------------------------
    const toHex = buffer => Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('');

    // The cloud reference changes the result, so its content is part of the output folder name
    async function referenceHash() {
        const refMode = document.querySelector('input[name="cloud-ref-mode"]:checked');
        const mode = refMode ? refMode.value : 'none';
        if (mode === 'none' || !(window.crypto && crypto.subtle)) return '';
        const parts = [];
        for (const slot of ({ pair: ['pencil', 'ink'], sheet: ['sheet'], style: ['style'] }[mode])) {
            const input = $('cloud-ref-' + slot);
            const file = input && input.files[0];
            if (!file) return '';
            try { parts.push(toHex(await crypto.subtle.digest('SHA-1', await file.arrayBuffer()))); }
            catch (e) { return ''; }
        }
        return toHex(await crypto.subtle.digest('SHA-1', new TextEncoder().encode(parts.join(':'))));
    }

    // Same fields the start buttons send, so the server names the folder the same way for the check and the run
    async function collect(mode) {
        if (mode === 'local') {
            const custom = $('custom-model-file');
            return {
                mode,
                model_name: $('process-model-select').value,
                custom_model_name: custom && custom.files[0] ? custom.files[0].name : '',
                patch_size: parseInt($('process-patch-size').value, 10),
                overlap: parseInt($('process-overlap').value, 10),
                contrast_scale: parseFloat($('process-contrast').value),
                upscale: parseFloat($('process-upscale').value),
                use_fp16: $('use-fp16').checked,
                use_preprocessed: $('process-use-pre').checked,
            };
        }
        const style = document.querySelector('input[name="cloud-style"]:checked');
        const refMode = document.querySelector('input[name="cloud-ref-mode"]:checked');
        return {
            mode,
            model: $('cloud-model-select').value,
            custom_model: $('cloud-custom-model').value.trim(),
            style: style ? style.value : 'fine',
            variants: $('cloud-variants').value,
            extra_prompt: $('cloud-extra-prompt').value,
            reference: refMode ? refMode.value : 'none',
            reference_hash: await referenceHash(),
            align: $('cloud-align').checked,
            use_preprocessed: $('cloud-use-pre').checked,
        };
    }

    async function refreshStatus(mode) {
        const token = ++statusTokens[mode];
        if (!current) return;
        try {
            const data = await api(projectUrl('/status'), jsonBody('POST', await collect(mode)));
            if (token !== statusTokens[mode]) return;   // a newer request is on its way
            status[mode] = data.drawings || {};
            labels[mode] = data.labels || null;
        } catch (e) {
            if (token !== statusTokens[mode]) return;
            status[mode] = {};
            labels[mode] = null;
        }
        if (pickers[mode]) pickers[mode].render();
        renderDestination(mode);
    }

    let statusTimers = {};
    function scheduleStatus(mode) {
        clearTimeout(statusTimers[mode]);
        statusTimers[mode] = setTimeout(() => refreshStatus(mode), 250);
    }

    function renderDestination(mode) {
        const node = $(mode === 'local' ? 'process-dest' : 'cloud-dest');
        if (!node) return;
        node.replaceChildren();
        if (!current || !labels[mode]) return;
        const usePre = $(mode === 'local' ? 'process-use-pre' : 'cloud-use-pre').checked;
        const folder = labels[mode][usePre ? 'preprocessed' : 'original'];
        node.append('Saved in ', el('code', null, `outputs/${labels[mode].original}`));
        if (usePre && folder !== labels[mode].original) node.append(' (', el('code', null, '…_pre'), ' for preprocessed drawings)');
        const external = current.project.settings && current.project.settings.external_dir;
        if (external) node.append(el('span', 'run-dest-extra', ` · also copied to ${external}`));
    }

    // ---- library picker -----------------------------------------------------------------
    // One component, five uses: the library in the Projects tab ("manage", no selection) and the
    // drawing choice of Preprocessing, Diagnostics, Batch and Cloud inking.
    const FILTERS = {
        manage: [['all', 'All'], ['pre', 'Preprocessed'], ['never', 'Not inked yet']],
        preprocess: [['notpre', 'Not preprocessed'], ['pre', 'Preprocessed'], ['all', 'All']],
        diag: [['all', 'All'], ['pre', 'Preprocessed']],
        local: [['todo', 'To ink with these settings'], ['done', 'Already inked'], ['all', 'All']],
        cloud: [['todo', 'To ink with these settings'], ['done', 'Already inked'], ['all', 'All']],
    };
    const EMPTY_FILTER = {
        notpre: 'Every drawing has a preprocessed version.',
        pre: 'No drawing has a preprocessed version yet.',
        never: 'Every drawing has been inked at least once.',
        todo: 'Every drawing is already inked with these settings. Change a setting, or show “Already inked” to run some again.',
        done: 'No drawing is inked with these settings yet.',
        all: 'No drawings.',
    };

    function createPicker(container) {
        const mode = container.dataset.mode;
        const selectable = mode !== 'manage';
        const max = mode === 'diag' ? 5 : Infinity;
        let filter = FILTERS[mode][0][0];
        let selected = new Set();
        let pristine = true;   // until the user ticks something, the selection follows the filter

        container.classList.add('picker');
        const bar = el('div', 'picker-bar');
        const filters = el('div', 'picker-filters');
        filters.setAttribute('role', 'group');
        filters.setAttribute('aria-label', 'Show');
        FILTERS[mode].forEach(([key, text]) => {
            const b = el('button', 'picker-filter', text);
            b.type = 'button';
            b.dataset.filter = key;
            b.addEventListener('click', () => {
                filter = key;
                pristine = true;
                render();
            });
            filters.appendChild(b);
        });
        const tools = el('div', 'picker-tools');
        const count = el('span', 'picker-count');
        count.setAttribute('aria-live', 'polite');
        tools.appendChild(count);
        let allBtn = null, noneBtn = null;
        if (selectable) {
            allBtn = el('button', 'picker-link', max < Infinity ? `Select first ${max}` : 'Select all');
            allBtn.type = 'button';
            allBtn.addEventListener('click', () => { pristine = false; selectVisible(); render(); });
            noneBtn = el('button', 'picker-link', 'Clear');
            noneBtn.type = 'button';
            noneBtn.addEventListener('click', () => { pristine = false; selected.clear(); render(); });
            tools.append(allBtn, noneBtn);
        }
        bar.append(filters, tools);
        const grid = el('div', 'picker-grid');
        const empty = el('div', 'picker-empty');
        container.replaceChildren(bar, grid, empty);

        grid.addEventListener('change', e => {
            const box = e.target.closest('.lib-check');
            if (!box) return;
            pristine = false;
            const name = box.closest('.lib-tile').dataset.name;
            if (box.checked) {
                if (selected.size >= max) { box.checked = false; flashCount(`At most ${max} drawings`); return; }
                selected.add(name);
            } else {
                selected.delete(name);
            }
            box.closest('.lib-tile').classList.toggle('is-selected', box.checked);
            updateCount();
        });
        grid.addEventListener('click', e => {
            const remove = e.target.closest('.lib-remove');
            if (remove) removeDrawing(remove.closest('.lib-tile').dataset.name);
        });

        function drawingState(d) {
            if (mode === 'local' || mode === 'cloud') return (status[mode][d.name] || {}).state || 'none';
            return null;
        }

        function matches(d) {
            switch (filter) {
                case 'pre': return d.preprocessed;
                case 'notpre': return !d.preprocessed;
                case 'never': return d.inked.length === 0;
                case 'todo': return drawingState(d) !== 'same';
                case 'done': return drawingState(d) === 'same';
                default: return true;
            }
        }

        function visible() {
            return current ? current.drawings.filter(matches) : [];
        }

        function selectVisible() {
            selected = new Set(visible().slice(0, max === Infinity ? undefined : max).map(d => d.name));
        }

        function versionShown(d) {
            if (mode === 'local' || mode === 'cloud') {
                const s = status[mode][d.name];
                if (s && s.version) return s.version;
            }
            const useBox = { diag: 'diag-use-pre', local: 'process-use-pre', cloud: 'cloud-use-pre' }[mode];
            return useBox && $(useBox).checked && d.preprocessed ? 'preprocessed' : 'original';
        }

        function tile(d) {
            const node = el(selectable ? 'label' : 'div', 'lib-tile');
            node.dataset.name = d.name;
            if (selectable) {
                const box = document.createElement('input');
                box.type = 'checkbox';
                box.className = 'lib-check';
                box.checked = selected.has(d.name);
                node.classList.toggle('is-selected', box.checked);
                node.appendChild(box);
            }
            const version = versionShown(d);
            const img = document.createElement('img');
            img.className = 'lib-thumb';
            img.loading = 'lazy';
            img.decoding = 'async';
            img.alt = '';
            const bust = version === 'preprocessed' ? (d.preprocessed_at || '') : (d.sha1 || '').slice(0, 8);
            img.src = projectUrl(`/thumb/${version}/${encodePath(d.name)}?v=${encodeURIComponent(bust)}`);
            img.addEventListener('error', () => img.classList.add('is-broken'), { once: true });
            node.appendChild(img);

            const meta = el('span', 'lib-meta');
            meta.appendChild(el('span', 'lib-name', d.name));
            const chips = el('span', 'lib-chips');
            const state = drawingState(d);
            if (state === 'same') chips.appendChild(el('span', 'chip chip-done', 'inked'));
            else if (state === 'other') chips.appendChild(el('span', 'chip chip-other', 'other settings'));
            if (d.preprocessed) chips.appendChild(el('span', 'chip chip-pre', version === 'preprocessed' || mode === 'manage' || mode === 'preprocess' ? 'preprocessed' : 'original used'));
            if (mode === 'manage' && d.inked.length) chips.appendChild(el('span', 'chip chip-done', `inked ×${d.inked.length}`));
            meta.appendChild(chips);
            node.appendChild(meta);

            const s = (mode === 'local' || mode === 'cloud') && status[mode][d.name];
            if (s && s.record) {
                node.title = `${s.state === 'same' ? 'Already inked with these settings' : 'Inked only with other settings'}, ${fmtDate(s.record.timestamp)}\n${describeParams(s.record.mode, s.record.params)}`;
            } else if (mode === 'manage' && d.inked.length) {
                node.title = `Inked with ${plural(d.inked.length, 'combination')} of settings:\n${d.inked.join('\n')}`;
            }
            if (mode === 'manage') {
                const remove = el('button', 'lib-remove');
                remove.type = 'button';
                remove.setAttribute('aria-label', `Remove ${d.name} from the project`);
                remove.appendChild(icon('trash3'));
                node.appendChild(remove);
            }
            return node;
        }

        let flashTimer = null;
        function flashCount(text) {
            count.textContent = text;
            count.classList.add('is-warning');
            clearTimeout(flashTimer);
            flashTimer = setTimeout(() => { count.classList.remove('is-warning'); updateCount(); }, 1600);
        }

        function updateCount() {
            const total = current ? current.drawings.length : 0;
            if (!selectable) {
                count.textContent = plural(total, 'drawing');
            } else {
                count.textContent = `${selected.size} of ${total} selected` + (max < Infinity ? ` · max ${max}` : '');
                if (allBtn) allBtn.disabled = visible().length === 0;
                if (noneBtn) noneBtn.disabled = selected.size === 0;
            }
            document.dispatchEvent(new CustomEvent('picker-change', { detail: { mode, count: selected.size } }));
        }

        function render() {
            filters.querySelectorAll('.picker-filter').forEach(b => {
                const on = b.dataset.filter === filter;
                b.classList.toggle('is-active', on);
                b.setAttribute('aria-pressed', on ? 'true' : 'false');
            });
            const names = new Set(current ? current.drawings.map(d => d.name) : []);
            selected = new Set([...selected].filter(n => names.has(n)));
            // "Already inked" never preselects: one click must not ink (or pay for) the same drawings again
            if (pristine && selectable && mode !== 'diag' && filter !== 'done') selectVisible();
            else if (pristine && filter === 'done') selected.clear();
            const shown = visible();
            grid.replaceChildren(...shown.map(tile));
            empty.replaceChildren();
            if (!current || current.drawings.length === 0) {
                empty.append(el('p', null, 'No drawings in this project yet.'));
                const add = el('button', 'btn btn-primary btn-small');
                add.type = 'button';
                add.append(icon('plus-lg'), ' Add drawings');
                add.addEventListener('click', () => $('library-files-input').click());
                empty.appendChild(add);
            } else if (shown.length === 0) {
                empty.append(el('p', null, EMPTY_FILTER[filter] || 'Nothing to show.'));
            }
            empty.hidden = empty.childNodes.length === 0;
            updateCount();
        }

        return {
            mode,
            render,
            selected: () => (current ? current.drawings.map(d => d.name).filter(n => selected.has(n)) : []),
            reset() { selected.clear(); pristine = true; filter = FILTERS[mode][0][0]; },
        };
    }

    // ---- library actions ------------------------------------------------------------------
    function importMessage(text, kind) {
        const box = $('library-import-message');
        box.textContent = text;
        box.className = `import-message is-${kind || 'info'}`;
        box.hidden = !text;
    }

    async function importFiles(fileList) {
        if (!current) return;
        const files = Array.from(fileList || []);
        const images = files.filter(f => IMAGE_RE.test(f.name));
        const ignored = files.length - images.length;
        if (!images.length) {
            if (files.length) importMessage('None of the chosen files is a PNG, JPG or TIFF drawing.', 'warning');
            return;
        }
        showProjectsTab();
        const totals = { added: 0, renamed: 0, duplicate: 0, skipped: ignored };
        try {
            for (let i = 0; i < images.length; i += UPLOAD_BATCH) {
                importMessage(`Adding drawings… ${Math.min(i + UPLOAD_BATCH, images.length)} of ${images.length}`, 'info');
                const form = new FormData();
                images.slice(i, i + UPLOAD_BATCH).forEach(f => form.append('files', f, f.name));
                const data = await api(projectUrl('/drawings'), { method: 'POST', body: form });
                Object.keys(totals).forEach(k => { totals[k] += (data.report[k] || []).length; });
            }
            const parts = [`${plural(totals.added + totals.renamed, 'drawing')} added`];
            if (totals.duplicate) parts.push(`${totals.duplicate} already in the project, not copied again`);
            if (totals.renamed) parts.push(`${totals.renamed} renamed because another drawing had the same name`);
            if (totals.skipped) parts.push(`${totals.skipped} skipped (not an image)`);
            await refresh();
            importMessage(parts.join(' · ') + '.', 'success');
        } catch (e) {
            await refresh();
            importMessage(`Adding drawings stopped: ${e.message}`, 'error');
        }
    }

    async function removeDrawing(name) {
        if (!window.confirm(`Remove ${name} from the project?\n\nIts inked results and the run history stay in the project folder.`)) return;
        try {
            await api(projectUrl(`/drawings/${encodePath(name)}`), { method: 'DELETE' });
            await refresh();
        } catch (e) {
            importMessage(e.message, 'error');
        }
    }

    // ---- project shell ----------------------------------------------------------------------
    function showProjectsTab() {
        const tab = document.querySelector('.tab-button[data-tab="projects-tab"]');
        if (tab && !tab.classList.contains('active')) tab.click();
    }

    function renderGates() {
        document.body.classList.toggle('has-project', !!current);
        document.querySelectorAll('[data-gate]').forEach(gate => {
            if (gate.childNodes.length) return;
            gate.append(icon('folder2-open'), el('p', 'gate-title', 'Open a project to continue'),
                el('p', 'gate-text', 'Drawings, results and settings live in a project. Create one, or open one you already have.'));
            const go = el('button', 'btn btn-primary');
            go.type = 'button';
            go.textContent = 'Go to projects';
            go.addEventListener('click', showProjectsTab);
            gate.appendChild(go);
        });
        document.querySelectorAll('[data-project-chip]').forEach(chip => {
            chip.replaceChildren();
            if (current) chip.append(icon('folder2-open'), ' ', current.name);
        });
        const badge = $('header-project-badge');
        badge.hidden = !current;
        $('header-project-name').textContent = current ? current.name : '';
        $('projects-home').hidden = !!current;
        $('project-workspace').hidden = !current;
    }

    function renderWorkspace() {
        if (!current) return;
        const p = current.project;
        $('project-title').textContent = p.project_name;
        $('project-desc').textContent = p.description || '';
        $('project-desc').hidden = !p.description;
        const runs = p.runs || [];
        const pre = current.drawings.filter(d => d.preprocessed).length;
        const facts = [plural(current.drawings.length, 'drawing')];
        if (pre) facts.push(`${pre} preprocessed`);
        facts.push(plural(runs.length, 'run'));
        if (runs.length) facts.push(`last run ${fmtDate(runs[runs.length - 1].timestamp)}`);
        $('project-facts').textContent = facts.join(' · ');
        $('project-external-dir').value = (p.settings && p.settings.external_dir) || '';
        externalStatus(p.settings && p.settings.external_dir
            ? 'Every run is also copied there, in a folder named after its settings.' : '', '');
        renderHistory(runs);
    }

    function renderHistory(runs) {
        const box = $('project-history');
        if (!runs.length) {
            box.replaceChildren(el('p', 'empty-note', 'No runs yet. Inked drawings and the settings used appear here.'));
            return;
        }
        box.replaceChildren(...runs.slice().reverse().map(runRow));
    }

    function runRow(run) {
        const items = run.items || [];
        const ok = items.filter(i => i.status === 'ok').length;
        const details = el('details', 'project-run');
        const summary = el('summary', 'project-run-summary');
        summary.appendChild(el('span', 'project-run-mode project-run-' + run.mode, run.mode === 'cloud' ? 'cloud' : 'local'));
        summary.appendChild(el('span', 'project-run-when', fmtDate(run.timestamp)));
        let text = `${ok} of ${plural(items.length, 'drawing')}`;
        if (run.mode === 'cloud' && run.cost_usd) text += ` · $${Number(run.cost_usd).toFixed(3)}`;
        summary.appendChild(el('span', 'project-run-count', text));
        details.appendChild(summary);

        const body = el('div', 'project-run-body');
        body.appendChild(el('p', 'project-run-params', describeParams(run.mode, run.params)));
        const folders = [...new Set(items.filter(i => i.outputs && i.outputs.length).map(i => i.config || run.config))];
        folders.forEach(f => body.appendChild(el('p', 'project-run-folder', `outputs/${f}`)));
        if (run.external) {
            const ext = run.external;
            body.appendChild(el('p', 'project-run-external' + (ext.error ? ' is-failed' : ''),
                ext.error ? `External copy failed (${ext.error}). The project copy is complete.`
                    : `Also copied: ${plural(ext.copied, 'file')} to ${ext.dir}`));
        }
        const failed = items.filter(i => i.status !== 'ok');
        if (failed.length) body.appendChild(el('p', 'project-run-failed', `Failed: ${failed.map(i => i.source).join(', ')}`));
        if (run.output_dir) {
            const open = el('button', 'btn btn-secondary btn-small');
            open.type = 'button';
            open.append(icon('folder2-open'), ' Open folder');
            open.addEventListener('click', () => { if (typeof openFolder === 'function') openFolder(run.output_dir); });
            body.appendChild(open);
        }
        details.appendChild(body);
        return details;
    }

    function externalStatus(text, kind) {
        const node = $('project-external-status');
        node.textContent = text;
        node.className = 'field-note' + (kind ? ` is-${kind}` : '');
    }

    async function saveExternal(value) {
        if (!current) return;
        try {
            const data = await api(projectUrl('/settings'), jsonBody('PATCH', { external_dir: value }));
            current.project.settings = data.settings;
            renderWorkspace();
            renderDestination('local');
            renderDestination('cloud');
        } catch (e) {
            externalStatus(e.message, 'error');
        }
    }

    // ---- home: list, create, open, close, delete ---------------------------------------------
    function homeMessage(text, type) {
        const box = $('project-message');
        box.replaceChildren(el('div', `message message-${type}`, text));
        box.style.display = 'block';
    }

    function projectRow(p) {
        const row = el('div', 'project-row');
        const main = el('button', 'project-row-main');
        main.type = 'button';
        main.appendChild(el('span', 'project-row-name', p.project_name));
        if (p.description) main.appendChild(el('span', 'project-row-desc', p.description));
        main.addEventListener('click', () => openProject(p.project_id).catch(e => homeMessage(e.message, 'error')));
        const facts = [plural(p.drawings_count || 0, 'drawing'), plural(p.runs_count, 'run')];
        facts.push(p.last_run ? `last run ${fmtDate(p.last_run)}` : `created ${fmtDate(p.created_at)}`);
        row.append(main, el('span', 'project-row-facts', facts.join(' · ')));
        const open = el('button', 'btn btn-secondary btn-small', 'Open');
        open.type = 'button';
        open.addEventListener('click', () => openProject(p.project_id).catch(e => homeMessage(e.message, 'error')));
        row.appendChild(open);
        return row;
    }

    async function loadList() {
        try {
            const projects = (await api('/api/projects')).projects || [];
            $('projects-list').replaceChildren(...projects.map(projectRow));
            $('projects-empty').style.display = projects.length ? 'none' : 'block';
        } catch (e) {
            homeMessage(e.message, 'error');
        }
    }

    async function load(id) {
        const [info, lib] = await Promise.all([
            api(`/api/projects/${encodeURIComponent(id)}`),
            api(`/api/projects/${encodeURIComponent(id)}/drawings`),
        ]);
        return { id, name: info.project.project_name, path: info.path, project: info.project, drawings: lib.drawings || [] };
    }

    async function openProject(id) {
        const next = await load(id);
        const switching = !current || current.id !== id;
        current = next;
        store(id);
        if (switching) {
            Object.values(pickers).forEach(p => p.reset());
            importMessage('', 'info');
            document.dispatchEvent(new CustomEvent('project-changed', { detail: { id } }));
        }
        renderGates();
        renderWorkspace();
        Object.values(pickers).forEach(p => p.render());
        await Promise.all([refreshStatus('local'), refreshStatus('cloud')]);
    }

    function closeProject() {
        current = null;
        store(null);
        status.local = {}; status.cloud = {};
        labels.local = null; labels.cloud = null;
        Object.values(pickers).forEach(p => { p.reset(); p.render(); });
        renderGates();
        document.dispatchEvent(new CustomEvent('project-changed', { detail: { id: null } }));
        loadList();
    }

    /** Reload library, history and "done" marks: after an import, a preprocessing or a run. */
    async function refresh() {
        if (!current) return;
        try {
            const next = await load(current.id);
            current.project = next.project;
            current.drawings = next.drawings;
            renderWorkspace();
            Object.values(pickers).forEach(p => p.render());
            await Promise.all([refreshStatus('local'), refreshStatus('cloud')]);
        } catch (e) { /* the run itself already succeeded */ }
    }

    async function createProject() {
        const name = $('project-name-input').value.trim();
        if (!name) { homeMessage('Give the project a name.', 'warning'); $('project-name-input').focus(); return; }
        try {
            const data = await api('/api/projects', jsonBody('POST', { name, description: $('project-desc-input').value.trim() }));
            $('project-name-input').value = '';
            $('project-desc-input').value = '';
            $('project-message').style.display = 'none';
            await openProject(data.project.project_id);
        } catch (e) {
            homeMessage(e.message, 'error');
        }
    }

    async function deleteProject() {
        if (!current) return;
        const ok = window.confirm(
            `Delete "${current.name}"?\n\nThis removes its drawings, every inked result and the run history from the project folder. ` +
            `Copies in an external folder are not touched.`);
        if (!ok) return;
        try {
            await api(projectUrl(''), { method: 'DELETE' });
            closeProject();
        } catch (e) {
            externalStatus(e.message, 'error');
        }
    }

    // ---- wiring -------------------------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', async () => {
        document.querySelectorAll('.library-picker').forEach(node => {
            pickers[node.dataset.mode] = createPicker(node);
        });

        $('project-create-btn').addEventListener('click', createProject);
        $('project-name-input').addEventListener('keydown', e => { if (e.key === 'Enter') createProject(); });
        $('project-close-btn').addEventListener('click', closeProject);
        $('project-delete-btn').addEventListener('click', deleteProject);
        $('project-open-folder-btn').addEventListener('click', () => {
            if (current && typeof openFolder === 'function') openFolder(current.path);
        });
        $('header-project-badge').addEventListener('click', showProjectsTab);

        $('library-add-files').addEventListener('click', () => $('library-files-input').click());
        $('library-add-folder').addEventListener('click', () => $('library-folder-input').click());
        ['library-files-input', 'library-folder-input'].forEach(id => {
            $(id).addEventListener('change', async e => {
                await importFiles(e.target.files);
                e.target.value = '';
            });
        });
        // drop files anywhere on the library
        const drop = $('project-workspace');
        drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('is-dropping'); });
        drop.addEventListener('dragleave', e => { if (!drop.contains(e.relatedTarget)) drop.classList.remove('is-dropping'); });
        drop.addEventListener('drop', e => {
            e.preventDefault();
            drop.classList.remove('is-dropping');
            importFiles(e.dataTransfer.files);
        });

        $('project-external-dir').addEventListener('change', e => saveExternal(e.target.value));
        $('project-external-browse').addEventListener('click', async () => {
            const browse = $('project-external-browse');
            browse.disabled = true;
            try {
                const data = await api('/api/select-directory', { method: 'POST' });
                if (data.directory) await saveExternal(data.directory);
            } catch (e) {
                externalStatus(e.message, 'error');
            } finally {
                browse.disabled = false;
            }
        });

        // any change to a setting may point to another output folder: re-check what is already inked
        [['processing-tab', 'local'], ['cloud-tab', 'cloud']].forEach(([id, mode]) => {
            const tab = $(id);
            const onChange = e => {
                if (!current || (e.target && e.target.id === 'cloud-api-key')) return;
                scheduleStatus(mode);
            };
            tab.addEventListener('input', onChange);
            tab.addEventListener('change', onChange);
        });
        $('diag-use-pre').addEventListener('change', () => pickers.diag && pickers.diag.render());

        renderGates();
        await loadList();
        const saved = stored();
        if (saved) {
            try { await openProject(saved); }
            catch (e) { store(null); renderGates(); }   // the project was deleted or moved
        }
    });

    // ---- re-inking: ask before running drawings again with the same settings ------------------
    // Resolves to the names to run (all, or only the new ones) or null when the person cancels.
    async function confirmRerun(mode, names) {
        await refreshStatus(mode);   // the settings may have changed since the last check
        const done = names.filter(n => (status[mode][n] || {}).state === 'same');
        if (!done.length) return names;
        const fresh = names.filter(n => !done.includes(n));
        const cloud = mode === 'cloud';

        return new Promise(resolve => {
            const overlay = el('div', 'modal rerun-dialog');
            overlay.setAttribute('role', 'dialog');
            overlay.setAttribute('aria-modal', 'true');
            overlay.setAttribute('aria-labelledby', 'rerun-title');
            const box = el('div', 'modal-content mac-window');
            const body = el('div', 'rerun-body');
            const title = el('h3', null, done.length === names.length
                ? (done.length === 1 ? 'This drawing is already inked with these settings' : `These ${done.length} drawings are already inked with these settings`)
                : `${done.length} of ${names.length} drawings are already inked with these settings`);
            title.id = 'rerun-title';
            const list = el('p', 'rerun-names', done.slice(0, 6).join(', ') + (done.length > 6 ? ` and ${done.length - 6} more` : ''));
            const note = el('p', 'rerun-note', `Inking them again replaces their results in the same folder${cloud ? ', and you pay for them again' : ''}.`);
            const actions = el('div', 'rerun-actions');

            function close(result) {
                document.removeEventListener('keydown', onKey);
                overlay.remove();
                resolve(result);
            }
            function onKey(e) { if (e.key === 'Escape') close(null); }
            function button(cls, iconName, label, result) {
                const b = el('button', `btn ${cls}`);
                b.type = 'button';
                b.append(icon(iconName), ` ${label}`);
                b.addEventListener('click', () => close(result));
                actions.appendChild(b);
                return b;
            }

            let first;
            if (fresh.length) {
                first = button('btn-primary', 'skip-forward-fill', `Ink only the ${fresh.length} new`, fresh);
                button('btn-secondary', 'arrow-repeat', `Ink all ${names.length} again`, names);
            } else {
                first = button('btn-primary', 'arrow-repeat', done.length === 1 ? 'Ink it again' : `Ink all ${done.length} again`, names);
            }
            button('btn-secondary', 'x-lg', 'Cancel', null);
            overlay.addEventListener('click', e => { if (e.target === overlay) close(null); });
            document.addEventListener('keydown', onKey);

            body.append(title, list, note, actions);
            box.appendChild(body);
            overlay.appendChild(box);
            document.body.appendChild(overlay);
            first.focus();
        });
    }

    window.PyProjects = {
        currentId: () => (current ? current.id : null),
        selected: mode => (pickers[mode] ? pickers[mode].selected() : []),
        confirmRerun,
        fileUrl: (area, rel) => (current ? projectUrl(`/file/${area}/${encodePath(rel)}`) : ''),
        refresh,
        referenceHash,
        refreshStatsList: () => document.dispatchEvent(new CustomEvent('stats-refresh')),
    };
})();
