// PyPotteryInk Flask App - Frontend JavaScript

// GPU-Accelerated Progress Setter (Impeccable 60fps performance without layout thrash)
function setProgress(el, percent) {
    if (!el) return;
    const clamped = Math.min(Math.max(percent, 0), 100);
    el.style.transform = `scaleX(${clamped / 100})`;
    el.style.width = '100%';
}

// Model Download Overlay Helper Functions
function showDownloadOverlay(message) {
    const overlay = document.getElementById('model-download-overlay');
    const messageEl = document.getElementById('download-message');
    const statusEl = document.getElementById('download-status');
    const fillEl = document.getElementById('download-progress-fill');

    if (message) messageEl.textContent = message;
    statusEl.textContent = 'Preparing download...';
    if (fillEl) setProgress(fillEl, 0);
    overlay.classList.add('visible');
}

function updateDownloadStatus(status) {
    const statusEl = document.getElementById('download-status');
    if (status) statusEl.textContent = status;
}

/** Drive the overlay's progress bar with a real percentage + message, from
 * an SSE update tagged 'style_model_download' or 'sdturbo_download'. */
function updateDownloadProgress(progress, message) {
    const fillEl = document.getElementById('download-progress-fill');
    const statusEl = document.getElementById('download-status');
    if (fillEl && progress !== undefined) {
        setProgress(fillEl, progress);
    }
    if (statusEl && message) statusEl.textContent = message;
}

function hideDownloadOverlay() {
    const overlay = document.getElementById('model-download-overlay');
    overlay.classList.remove('visible');
}

document.addEventListener('DOMContentLoaded', function () {
    const splashScreen = document.getElementById('splash-screen');
    const mainContainer = document.getElementById('main-container');
    const progressBar = document.getElementById('splash-progress-bar');
    const progressText = document.getElementById('splash-progress-text');
    const splashMessage = document.getElementById('splash-message');

    let progress = 0;
    const totalSteps = 5;

    // Simulate loading steps
    const loadingSteps = [
        { progress: 20, message: 'Loading models configuration...' },
        { progress: 40, message: 'Initializing hardware check...' },
        { progress: 60, message: 'Setting up processing environment...' },
        { progress: 80, message: 'Preparing user interface...' },
        { progress: 100, message: 'Ready!' }
    ];

    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < loadingSteps.length) {
            const step = loadingSteps[currentStep];
            progress = step.progress;
            setProgress(progressBar, progress);
            progressText.textContent = progress + '%';
            splashMessage.textContent = step.message;
            currentStep++;
        } else {
            clearInterval(stepInterval);
            setTimeout(() => {
                splashScreen.classList.add('fade-out');
                setTimeout(() => {
                    splashScreen.style.display = 'none';
                    mainContainer.style.display = 'block';
                }, 500);
            }, 300);
        }
    }, 400);
});

// Tab Navigation
document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', function () {
        const targetTab = this.getAttribute('data-tab');

        // Remove active class from all tabs and buttons
        document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Add active class to clicked button and corresponding tab
        this.classList.add('active');
        document.getElementById(targetTab).classList.add('active');
    });
});

// Hardware Check
document.getElementById('check-hardware-btn').addEventListener('click', async function () {
    const reportContainer = document.getElementById('hardware-report');
    const btn = this;

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Checking...';
    reportContainer.style.display = 'none';

    try {
        const response = await fetch('/api/hardware-check');
        const data = await response.json();

        if (data.success) {
            const report = data.report;
            const suitability = report.suitability;
            const components = report.components;
            const tips = report.tips || [];

            // Build HTML
            let html = '<div class="hardware-report">';

            // Suitability badge
            const suitabilityIcon = suitability.icon_class ? `<i class="bi ${suitability.icon_class}"></i>` : '';
            html += `<div class="suitability-badge suitability-${suitability.level}">
                <span class="badge-icon">${suitabilityIcon}</span>
                <span>${suitability.label}</span>
            </div>`;

            // Conclusion
            html += `<p class="suitability-conclusion">${suitability.conclusion}</p>`;

            // Component cards
            html += '<div class="hardware-cards">';
            for (const [key, comp] of Object.entries(components)) {
                const statusIcon = comp.status === 'excellent'
                    ? '<i class="bi bi-check-circle-fill"></i>'
                    : (comp.status === 'adequate'
                        ? '<i class="bi bi-exclamation-triangle-fill"></i>'
                        : '<i class="bi bi-x-circle-fill"></i>');
                const compIcon = comp.icon_class ? `<i class="bi ${comp.icon_class}"></i>` : '';
                html += `<div class="hardware-card">
                    <div class="hardware-card-header">
                        <span class="hardware-card-icon">${compIcon}</span>
                        <span class="hardware-card-title">${comp.name}</span>
                    </div>
                    <div class="hardware-card-value">${comp.value}</div>
                    <div class="hardware-card-status status-${comp.status}">
                        ${statusIcon} <span>${comp.message}</span>
                    </div>
                </div>`;
            }
            html += '</div>';

            // Tips
            if (tips.length > 0) {
                html += '<div class="hardware-tips"><h4><i class="bi bi-lightbulb-fill"></i> Tips</h4><ul>';
                tips.forEach(tip => {
                    html += `<li>${tip}</li>`;
                });
                html += '</ul></div>';
            }

            html += '</div>';

            reportContainer.innerHTML = html;
            reportContainer.style.display = 'block';
        } else {
            showMessage('error', 'Hardware check failed: ' + data.error, reportContainer);
        }
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, reportContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-search"></i> Check Hardware';
    }
});

// Follow a background run over Server-Sent Events. Model downloads (style .pkl or the shared sd-turbo
// backbone) drive the download overlay; any other message means the run itself is progressing.
function followRun(sessionId, { onProgress, onDone, onError }) {
    const eventSource = new EventSource(`/api/progress/${sessionId}`);
    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);
        if (data.keepalive) return;
        if (data.type === 'style_model_download' || data.type === 'sdturbo_download') {
            showDownloadOverlay(data.type === 'sdturbo_download'
                ? 'Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).'
                : 'Downloading style model...');
            updateDownloadProgress(data.progress, data.message);
            return;
        }
        hideDownloadOverlay();
        if (data.error) {
            eventSource.close();
            onError(data.message || 'The run stopped with an error.');
            return;
        }
        if (onProgress) onProgress(data);
        if (data.completed && data.results) {
            eventSource.close();
            onDone(data.results);
        }
    };
    eventSource.onerror = function () {
        eventSource.close();
        onError('Connection to the app lost. The run may still finish in the background: its results will appear in the project.');
    };
    return eventSource;
}

// Upload the custom .pkl chosen in a tab (kept server-side for the session)
async function uploadCustomModel(inputId) {
    const file = document.getElementById(inputId).files[0];
    if (!file) throw new Error('Choose the custom model file (.pkl) first.');
    const form = new FormData();
    form.append('model', file);
    const response = await fetch('/api/upload-custom-model', { method: 'POST', body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.success) throw new Error(data.error || 'Custom model upload failed');
}

// Primary button of a work tab: its label says what will happen, the hint says what is missing
function setRunButton(btnId, hintId, { count, verb, noun, missing, busy }) {
    const btn = document.getElementById(btnId);
    const hint = document.getElementById(hintId);
    const label = btn.querySelector('.btn-run-label');
    btn.disabled = !!busy || count === 0 || (missing && missing.length > 0);
    if (label && !busy) label.textContent = count ? `${verb} ${count} ${noun}${count === 1 ? '' : 's'}` : `${verb} ${noun}s`;
    if (hint) hint.textContent = busy ? '' : (count === 0 ? `Select ${noun}s from the project to continue.` :
        (missing && missing.length ? `To continue: ${missing.join(', ')}.` : ''));
}

const runBusy = { local: false, diag: false, preprocess: false };
const runCounts = { local: 0, diag: 0, preprocess: 0 };

function refreshRunButtons() {
    setRunButton('process-images-btn', 'process-hint', { count: runCounts.local, verb: 'Ink', noun: 'drawing', busy: runBusy.local });
    setRunButton('run-diagnostics-btn', 'diag-hint', { count: runCounts.diag, verb: 'Test', noun: 'drawing', busy: runBusy.diag });
    const stats = document.getElementById('stats-select');
    setRunButton('preprocess-btn', 'preprocess-hint', {
        count: runCounts.preprocess, verb: 'Preprocess', noun: 'drawing', busy: runBusy.preprocess,
        missing: stats && stats.value ? [] : ['a statistics file']
    });
    document.getElementById('calculate-stats-btn').disabled = runBusy.preprocess || runCounts.preprocess === 0;
}

document.addEventListener('picker-change', e => {
    if (e.detail.mode in runCounts) {
        runCounts[e.detail.mode] = e.detail.count;
        refreshRunButtons();
    }
});

// Results on screen belong to the previous project: hide them when another one opens
document.addEventListener('project-changed', () => {
    ['process-output', 'process-gallery', 'process-progress', 'diagnostics-output', 'diagnostics-gallery',
        'diag-progress', 'preprocess-output', 'preprocess-progress', 'stats-output', 'cloud-output', 'cloud-gallery']
        .forEach(id => {
            const node = document.getElementById(id);
            if (node) node.style.display = 'none';
        });
    loadStatsOptions();
});

// Diagnostics
document.getElementById('run-diagnostics-btn').addEventListener('click', async function () {
    const drawings = PyProjects.selected('diag');
    const modelName = document.getElementById('diag-model-select').value;
    const outputContainer = document.getElementById('diagnostics-output');
    const gallery = document.getElementById('diagnostics-gallery');
    const progressContainer = document.getElementById('diag-progress');
    const progressFill = document.getElementById('diag-progress-fill');
    const statusText = document.getElementById('diag-status');
    const btn = this;

    const finish = () => {
        runBusy.diag = false;
        btn.querySelector('.btn-run-label').textContent = '';
        refreshRunButtons();
    };
    runBusy.diag = true;
    refreshRunButtons();
    btn.querySelector('.btn-run-label').textContent = 'Running…';
    outputContainer.style.display = 'none';
    gallery.style.display = 'none';
    gallery.innerHTML = '';

    try {
        if (modelName === 'custom') await uploadCustomModel('diag-custom-model-file');

        const diffusionCheck = await (await fetch('/api/check-diffusion-model')).json();
        if (!diffusionCheck.cached) {
            showDownloadOverlay('Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).');
        }

        const response = await fetch('/api/diagnostics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: PyProjects.currentId(),
                drawings,
                use_preprocessed: document.getElementById('diag-use-pre').checked,
                model_name: modelName,
                patch_size: parseInt(document.getElementById('diag-patch-size').value),
                overlap: parseInt(document.getElementById('diag-overlap').value),
                contrast_values: document.getElementById('diag-contrast').value
            })
        });
        const resp = await response.json();
        if (!resp.success) throw new Error(resp.error || 'Could not start the diagnostics');

        progressContainer.style.display = 'block';
        setProgress(progressFill, 5);
        statusText.textContent = 'Starting diagnostics...';

        followRun(resp.session_id, {
            onProgress(data) {
                if (data.progress !== undefined) setProgress(progressFill, data.progress);
                if (data.message) statusText.textContent = data.message;
            },
            onDone(results) {
                setProgress(progressFill, 100);
                statusText.textContent = 'Completed';
                showDiagnosticsComplete(results, outputContainer, gallery);
                finish();
            },
            onError(message) {
                showMessage('error', escapeHtml(message), outputContainer);
                progressContainer.style.display = 'none';
                finish();
            }
        });
    } catch (error) {
        hideDownloadOverlay();
        showMessage('error', escapeHtml(error.message), outputContainer);
        finish();
    }
});

function showDiagnosticsComplete(results, outputContainer, gallery) {
    const files = results.diagnostic_files || [];
    lastDiagnosticsDir = results.output_dir;
    outputContainer.innerHTML = `<div class="message message-success"><i class="bi bi-check-circle-fill"></i> <span>${files.length} diagnostic images, saved in the project.</span></div>
        <button class="btn btn-open-folder" type="button" onclick="openDiagnosticsFolder()"><i class="bi bi-folder2-open"></i> Open folder</button>`;
    outputContainer.style.display = 'block';

    gallery.innerHTML = '';
    files.forEach(file => {
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.decoding = 'async';
        img.src = PyProjects.fileUrl('diagnostics', file);
        img.alt = file.split('/').pop();
        img.onclick = function () { openLightbox(this.src); };
        gallery.appendChild(img);
    });
    gallery.style.display = 'grid';
}

// Open a folder in the system file manager (same route for Diagnostics and Batch Processing)
async function openFolder(folderPath) {
    try {
        await fetch('/api/open-folder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: folderPath })
        });
    } catch (error) {
        console.error('Failed to open folder:', error);
    }
}

// Folders of the last runs (kept in variables: a Windows path does not survive an inline onclick)
let lastDiagnosticsDir = null;
function openDiagnosticsFolder() {
    if (lastDiagnosticsDir) return openFolder(lastDiagnosticsDir);
}

let lastBatchOutputDir = null;
function openBatchOutputFolder() {
    if (lastBatchOutputDir) return openFolder(lastBatchOutputDir);
}

// Lightbox functions
function openLightbox(src) {
    const overlay = document.getElementById('lightbox-overlay');
    const img = document.getElementById('lightbox-image');
    img.src = src;
    overlay.classList.add('active');
}

function closeLightbox() {
    const overlay = document.getElementById('lightbox-overlay');
    overlay.classList.remove('active');
}

// Lightbox event listeners
document.addEventListener('DOMContentLoaded', function () {
    const lightboxOverlay = document.getElementById('lightbox-overlay');
    const lightboxClose = document.getElementById('lightbox-close');

    if (lightboxClose) {
        lightboxClose.addEventListener('click', closeLightbox);
    }

    if (lightboxOverlay) {
        lightboxOverlay.addEventListener('click', function (e) {
            if (e.target === lightboxOverlay) {
                closeLightbox();
            }
        });
    }

    // Close on Escape key
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });
});

// Statistics files offered by Preprocessing: the project's own, the ones shipped with the app, or a new .npy
async function loadStatsOptions(selectName) {
    const select = document.getElementById('stats-select');
    const projectId = PyProjects.currentId();
    const previous = selectName || select.value;
    select.replaceChildren();
    const placeholder = new Option('Choose statistics…', '');
    select.appendChild(placeholder);
    if (projectId) {
        try {
            const data = await (await fetch(`/api/projects/${encodeURIComponent(projectId)}/stats`)).json();
            const addGroup = (label, source, names) => {
                if (!names || !names.length) return;
                const group = document.createElement('optgroup');
                group.label = label;
                names.forEach(n => group.appendChild(new Option(n.replace(/\.npy$/, '').replace(/_stats$/, '').replace(/_/g, ' '), `${source}:${n}`)));
                select.appendChild(group);
            };
            addGroup('This project', 'project', data.project);
            addGroup('Shipped with the app', 'bundled', data.bundled);
        } catch (e) { /* the upload option below still works */ }
    }
    select.appendChild(new Option('Add a statistics file (.npy)…', 'upload'));
    const values = Array.from(select.options).map(o => o.value);
    select.value = previous && previous !== 'upload' && values.includes(previous) ? previous : (values.find(v => v.startsWith('bundled:')) || '');
    refreshRunButtons();
}

document.addEventListener('stats-refresh', () => loadStatsOptions());

document.getElementById('stats-select').addEventListener('change', function () {
    if (this.value === 'upload') {
        this.value = '';
        document.getElementById('stats-file-upload').click();
    }
    refreshRunButtons();
});

document.getElementById('stats-file-upload').addEventListener('change', async function () {
    const file = this.files[0];
    this.value = '';
    if (!file) return;
    const output = document.getElementById('preprocess-output');
    try {
        const form = new FormData();
        form.append('stats_file', file);
        form.append('project_id', PyProjects.currentId());
        const response = await fetch('/api/upload-stats', { method: 'POST', body: form });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Statistics file upload failed');
        await loadStatsOptions(`project:${data.stats_name}`);
    } catch (error) {
        showMessage('error', escapeHtml(error.message), output);
    }
});

// Calculate Statistics from the drawings selected in Preprocessing
document.getElementById('calculate-stats-btn').addEventListener('click', async function () {
    const btn = this;
    const outputContainer = document.getElementById('stats-output');
    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Calculating...';
    outputContainer.style.display = 'none';

    try {
        const response = await fetch('/api/calculate-statistics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: PyProjects.currentId(),
                drawings: PyProjects.selected('preprocess'),
                name: document.getElementById('stats-save-name').value.trim()
            })
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Statistics calculation failed');
        outputContainer.textContent = `Saved as ${data.summary.statistics_file}, from ${data.summary.images_analyzed} drawings. It is now selected above.`;
        outputContainer.style.display = 'block';
        await loadStatsOptions(`project:${data.summary.statistics_file}`);
    } catch (error) {
        outputContainer.textContent = error.message;
        outputContainer.style.display = 'block';
    } finally {
        btn.innerHTML = '<i class="bi bi-bar-chart-line"></i> Calculate';
        refreshRunButtons();
    }
});

// Preprocessing: makes the preprocessed version of each selected drawing inside the project
document.getElementById('preprocess-btn').addEventListener('click', async function () {
    const btn = this;
    const outputContainer = document.getElementById('preprocess-output');
    const progressContainer = document.getElementById('preprocess-progress');
    const progressFill = document.getElementById('preprocess-progress-fill');
    const statusText = document.getElementById('preprocess-status');
    const [source, ...nameParts] = document.getElementById('stats-select').value.split(':');

    const finish = () => { runBusy.preprocess = false; refreshRunButtons(); };
    runBusy.preprocess = true;
    refreshRunButtons();
    btn.querySelector('.btn-run-label').textContent = 'Preprocessing…';
    outputContainer.style.display = 'none';

    try {
        const response = await fetch('/api/preprocess-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: PyProjects.currentId(),
                drawings: PyProjects.selected('preprocess'),
                stats_source: source,
                stats_name: nameParts.join(':')
            })
        });
        const resp = await response.json();
        if (!resp.success) throw new Error(resp.error || 'Could not start preprocessing');

        progressContainer.style.display = 'block';
        setProgress(progressFill, 3);
        statusText.textContent = 'Checking drawings...';

        followRun(resp.session_id, {
            onProgress(data) {
                if (data.progress !== undefined) setProgress(progressFill, data.progress);
                if (data.message) statusText.textContent = data.message;
            },
            onDone(results) {
                progressContainer.style.display = 'none';
                const parts = [`<strong>${results.adjusted}</strong> corrected`];
                if (results.unchanged.length) parts.push(`<strong>${results.unchanged.length}</strong> already in range (they keep using the original)`);
                let html = `<div class="message message-success"><i class="bi bi-check-circle-fill"></i> <span>${parts.join(' · ')}, with ${escapeHtml(results.stats)}.</span></div>`;
                if (results.failed.length) {
                    html += `<div class="message message-warning"><i class="bi bi-exclamation-triangle-fill"></i> <span>Could not read: ${results.failed.map(f => escapeHtml(f.name)).join(', ')}</span></div>`;
                }
                outputContainer.innerHTML = html;
                outputContainer.style.display = 'block';
                PyProjects.refresh();
                finish();
            },
            onError(message) {
                progressContainer.style.display = 'none';
                showMessage('error', escapeHtml(message), outputContainer);
                finish();
            }
        });
    } catch (error) {
        showMessage('error', escapeHtml(error.message), outputContainer);
        finish();
    }
});

// Batch inking with the local model: the run takes the whole screen until it ends or is stopped
const localRun = (() => {
    const $ = id => document.getElementById(id);
    const modal = $('local-modal');
    let sessionId = null, finished = true, startedAt = 0, total = 0, fraction = 0, fractionAt = 0, timer = null;

    function fmtClock(sec) {
        sec = Math.max(0, Math.round(sec));
        const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = String(sec % 60).padStart(2, '0');
        return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
    }
    function tick() {
        $('local-stat-time').textContent = fmtClock((Date.now() - startedAt) / 1000);
        if (!finished && fraction > 0.02) {
            // time per unit of work so far, applied to what is left
            const left = (fractionAt - startedAt) / 1000 / fraction * (1 - fraction) - (Date.now() - fractionAt) / 1000;
            $('local-stat-eta').textContent = left > 1 ? `about ${fmtClock(left)} left` : 'almost done';
        }
    }
    function message(kind, html) {
        const box = $('local-run-message');
        box.className = `cloud-run-message is-${kind}`;
        box.innerHTML = html;
        box.style.display = 'block';
    }
    function open(count) {
        finished = false; sessionId = null; total = count; fraction = 0;
        $('local-run-status').textContent = `Inking ${count} drawing${count === 1 ? '' : 's'}`;
        $('local-run-current').textContent = 'Preparing...';
        setProgress($('local-run-fill'), 0);
        setProgress($('local-patch-fill'), 0);
        $('local-patch-status').textContent = '';
        document.querySelector('#local-modal .run-screen-patch').hidden = false;
        $('local-run-bar').setAttribute('aria-valuenow', '0');
        $('local-stat-done').textContent = `0 / ${count}`;
        $('local-stat-eta').textContent = '';
        $('local-run-message').style.display = 'none';
        const stop = $('local-stop-btn');
        stop.style.display = '';
        stop.disabled = true;   // enabled once the server has the run
        $('local-open-folder-btn').style.display = 'none';
        $('local-view-btn').style.display = 'none';
        $('local-modal-close').style.display = 'none';
        modal.style.display = 'flex';
        startedAt = Date.now();
        clearInterval(timer);
        timer = setInterval(tick, 1000);
        tick();
    }
    function started(id) {
        sessionId = id;
        $('local-stop-btn').disabled = false;
    }
    function progress(data) {
        if (finished) return;
        if (data.message) $('local-run-current').textContent = data.message;
        if (data.patch_progress !== undefined) {
            setProgress($('local-patch-fill'), data.patch_progress);
            $('local-patch-status').textContent = data.patch_message || '';
        }
        if (data.progress !== undefined) {
            setProgress($('local-run-fill'), data.progress);
            $('local-run-bar').setAttribute('aria-valuenow', String(Math.round(data.progress)));
            $('local-stat-done').textContent = `${Math.min(total, Math.floor(data.progress * total / 100 + 1e-6))} / ${total}`;
            // the patch bar refines the estimate inside the drawing being inked
            const within = data.patch_progress !== undefined && data.progress < 100 ? data.patch_progress / 100 / total : 0;
            fraction = Math.min(1, data.progress / 100 + within);
            fractionAt = Date.now();
        }
    }
    function end() {
        finished = true;
        clearInterval(timer);
        tick();
        $('local-stat-eta').textContent = '';
        document.querySelector('#local-modal .run-screen-patch').hidden = true;
        $('local-stop-btn').style.display = 'none';
        $('local-modal-close').style.display = '';
    }
    function done(results) {
        end();
        const notStarted = results.not_started || 0;
        $('local-run-status').textContent = notStarted ? 'Stopped' : 'Done';
        $('local-run-current').textContent = '';
        $('local-stat-done').textContent = `${results.successful} / ${total}`;
        setProgress($('local-run-fill'), 100 * (total - notStarted) / total);
        const failed = results.failed ? ` ${results.failed} failed.` : '';
        const stopped = notStarted ? ` ${notStarted} not started: they are still in “To ink”.` : '';
        const reasons = (results.errors || []).map(r => `<br><code>${escapeHtml(r)}</code>`).join('');
        if (!results.successful && results.failed) {
            $('local-run-status').textContent = 'Nothing was inked';
            message('error', `All ${results.failed} drawings failed.${stopped} The error:${reasons}`);
        } else {
            message(results.failed || notStarted ? 'warning' : 'success',
                `${results.successful} drawing${results.successful === 1 ? '' : 's'} inked.${failed}${stopped}${results.failed ? reasons : ''}`);
        }
        $('local-open-folder-btn').style.display = results.successful ? '' : 'none';
        const view = $('local-view-btn');
        view.style.display = results.successful ? '' : 'none';
        if (results.successful) view.focus();
    }
    function fail(text) {
        end();
        $('local-run-status').textContent = 'The run stopped';
        $('local-run-current').textContent = '';
        message('error', escapeHtml(text));
    }
    function close() {
        if (!finished) return;
        modal.style.display = 'none';
    }

    $('local-stop-btn').addEventListener('click', async function () {
        if (!sessionId || finished) return;
        this.disabled = true;
        $('local-run-status').textContent = 'Stopping...';
        $('local-run-current').textContent = 'Finishing the drawing being inked; the ones done so far are kept';
        try { await fetch(`/api/process-images/stop/${sessionId}`, { method: 'POST' }); } catch (e) { /* the run ends on its own */ }
    });
    $('local-open-folder-btn').addEventListener('click', openBatchOutputFolder);
    $('local-view-btn').addEventListener('click', () => {
        close();
        document.getElementById('process-output').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    $('local-modal-close').addEventListener('click', close);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && modal.style.display === 'flex') close(); });

    return { open, started, progress, done, fail };
})();

document.getElementById('process-images-btn').addEventListener('click', async function () {
    if (runBusy.local) return;
    const drawings = await PyProjects.confirmRerun('local', PyProjects.selected('local'));
    if (!drawings || !drawings.length) return;
    const modelName = document.getElementById('process-model-select').value;
    const btn = this;
    const outputContainer = document.getElementById('process-output');
    const gallery = document.getElementById('process-gallery');
    const status = text => localRun.progress({ message: text });

    const finish = () => { runBusy.local = false; refreshRunButtons(); };
    runBusy.local = true;
    refreshRunButtons();
    btn.querySelector('.btn-run-label').textContent = 'Inking…';
    outputContainer.style.display = 'none';
    gallery.style.display = 'none';
    localRun.open(drawings.length);

    try {
        if (modelName === 'custom') {
            status('Uploading custom model...');
            await uploadCustomModel('custom-model-file');
        } else {
            status('Checking model...');
            const modelCheck = await (await fetch('/api/check-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            })).json();
            if (!modelCheck.success) throw new Error('Model not available: ' + modelCheck.error);
        }

        status('Checking diffusion model...');
        const diffusionCheck = await (await fetch('/api/check-diffusion-model')).json();
        if (!diffusionCheck.cached) {
            showDownloadOverlay('Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).');
        }

        status('Loading the model...');
        const customFile = document.getElementById('custom-model-file').files[0];
        const response = await fetch('/api/process-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_id: PyProjects.currentId(),
                drawings,
                use_preprocessed: document.getElementById('process-use-pre').checked,
                model_name: modelName,
                custom_model_name: modelName === 'custom' && customFile ? customFile.name : '',
                use_fp16: document.getElementById('use-fp16').checked,
                contrast_scale: parseFloat(document.getElementById('process-contrast').value),
                patch_size: parseInt(document.getElementById('process-patch-size').value),
                overlap: parseInt(document.getElementById('process-overlap').value),
                upscale: parseFloat(document.getElementById('process-upscale').value)
            })
        });
        const processData = await response.json();
        if (!processData.success) throw new Error(processData.error || 'Could not start processing');
        localRun.started(processData.session_id);

        followRun(processData.session_id, {
            onProgress: localRun.progress,
            onDone(results) {
                renderBatchResults(results, outputContainer, gallery);
                localRun.done(results);
                PyProjects.refresh();   // the run is now in the project history, and the drawings are marked inked
                finish();
            },
            onError(message) {
                localRun.fail(message);
                showMessage('error', escapeHtml(message), outputContainer);
                finish();
            }
        });
    } catch (error) {
        hideDownloadOverlay();
        localRun.fail(error.message);
        showMessage('error', escapeHtml(error.message), outputContainer);
        finish();
    }
});

function renderBatchResults(results, outputContainer, gallery) {
    const failed = results.failed ? ` · <strong>${results.failed}</strong> failed` : '';
    const stopped = results.not_started ? ` · stopped, <strong>${results.not_started}</strong> not started` : '';
    const reasons = (results.errors || []).map(r => `<br><code>${escapeHtml(r)}</code>`).join('');
    let html = results.successful
        ? `<div class="message ${results.failed ? 'message-warning' : 'message-success'}"><i class="bi bi-check-circle-fill"></i> <span><strong>${results.successful}</strong> inked${failed}${stopped} · ${results.average_time.toFixed(1)} s per drawing${results.failed ? reasons : ''}</span></div>`
        : `<div class="message message-error"><i class="bi bi-x-circle-fill"></i> <span>Nothing was inked: <strong>${results.failed}</strong> failed.${reasons}</span></div>`;
    html += '<dl class="result-facts">';
    (results.folders || []).forEach(f => { html += `<dt>Folder</dt><dd><code>outputs/${escapeHtml(f)}</code></dd>`; });
    if (results.external) {
        html += results.external.error
            ? `<dt>External copy</dt><dd class="is-failed">failed (${escapeHtml(results.external.error)}); the results are complete in the project</dd>`
            : `<dt>External copy</dt><dd>${results.external.copied} files in <code>${escapeHtml(results.external.dir)}</code></dd>`;
    }
    html += '</dl>';
    lastBatchOutputDir = results.output_dir;
    if (results.successful) html += `<button class="btn btn-primary btn-view-results" type="button" onclick="openBatchOutputFolder()"><i class="bi bi-folder2-open"></i> Open the results folder</button>`;
    outputContainer.innerHTML = html;
    outputContainer.style.display = 'block';

    const comparisons = results.comparison_images || [];
    gallery.replaceChildren();
    if (!comparisons.length) { gallery.style.display = 'none'; return; }
    const title = document.createElement('h3');
    title.className = 'group-title';
    title.textContent = 'Pencil and ink, side by side';
    const grid = document.createElement('div');
    grid.className = 'gallery';
    comparisons.slice(0, 20).forEach(rel => {
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.decoding = 'async';
        img.src = PyProjects.fileUrl('outputs', rel);
        img.alt = rel.split('/').pop();
        img.onclick = function () { openLightbox(this.src); };
        grid.appendChild(img);
    });
    gallery.append(title, grid);
    if (comparisons.length > 20) {
        const note = document.createElement('p');
        note.className = 'field-note';
        note.textContent = `Showing 20 of ${comparisons.length}. All of them are in the folder.`;
        gallery.appendChild(note);
    }
    gallery.style.display = 'block';
}

function escapeHtml(text) {
    const node = document.createElement('div');
    node.textContent = String(text);
    return node.innerHTML;
}

function showMessage(type, message, container) {
    const messageClass = `message message-${type}`;
    const icons = {
        success: '<i class="bi bi-check-circle-fill"></i>',
        error: '<i class="bi bi-x-circle-fill"></i>',
        warning: '<i class="bi bi-exclamation-triangle-fill"></i>',
        info: '<i class="bi bi-info-circle-fill"></i>'
    };
    const icon = icons[type] || icons.info;
    container.innerHTML = `<div class="${messageClass}">${icon} <span>${message}</span></div>`;
    container.style.display = 'block';
}

// Copy Citation Helper Function
window.copyCitation = function (elementId = 'citation-text', btnElement) {
    const textEl = document.getElementById(elementId);
    if (!textEl) return;
    const text = (textEl.innerText || textEl.textContent).replace(/^"|"$/g, '').trim();
    navigator.clipboard.writeText(text).then(() => {
        const btn = btnElement || (window.event && window.event.target ? window.event.target.closest('.mac-copy-link') : null) || document.querySelector('.mac-copy-link');
        if (btn) {
            const orig = btn.innerHTML;
            btn.innerHTML = '<i class="bi bi-check2"></i> Copied!';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerHTML = orig;
                btn.classList.remove('copied');
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy citation:', err);
    });
};

// Fetch System Hardware Information
function fetchSystemInfo() {
    const cpuEl = document.getElementById('systemCPU');
    const gpuEl = document.getElementById('systemGPU');
    if (!cpuEl || !gpuEl) return;

    fetch('/api/system-info')
        .then(res => res.json())
        .then(data => {
            const cores = (data.cpu && data.cpu.cores) || data.cpu_count || 1;
            const platform = (data.cpu && data.cpu.platform) || data.platform || '';
            cpuEl.innerHTML = `<i class="bi bi-cpu me-1"></i> ${cores} Cores${platform ? ` (${platform})` : ''}`;

            const cuda = (data.gpu && data.gpu.cuda_available) || data.cuda_available;
            const gpuNames = (data.gpu && data.gpu.gpu_names) || (data.cuda_device_name ? [data.cuda_device_name] : []);
            const mps = (data.mps && data.mps.mps_available) || data.mps_available;

            if (cuda) {
                const name = gpuNames.length > 0 ? gpuNames[0] : 'NVIDIA CUDA';
                gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> ${name} (CUDA)`;
                gpuEl.className = 'chip-active';
            } else if (mps) {
                gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> Apple Silicon (MPS)`;
                gpuEl.className = 'chip-active';
            } else {
                gpuEl.innerHTML = `<i class="bi bi-gpu-card me-1"></i> CPU Only`;
                gpuEl.className = 'chip-cpu-only';
            }
        })
        .catch(err => {
            console.error('Failed to load system info:', err);
            cpuEl.innerHTML = '<i class="bi bi-cpu me-1"></i> Available';
            gpuEl.innerHTML = '<i class="bi bi-gpu-card me-1"></i> CPU Only';
            gpuEl.className = 'chip-cpu-only';
        });
}

// Info Modal
const infoBtnEl = document.getElementById('info-btn');
const infoModalEl = document.getElementById('info-modal');
const closeInfoModalEl = document.getElementById('close-info-modal');

if (infoBtnEl && infoModalEl) {
    infoBtnEl.addEventListener('click', function () {
        infoModalEl.style.display = 'flex';
        fetchSystemInfo();
    });
}

if (closeInfoModalEl && infoModalEl) {
    closeInfoModalEl.addEventListener('click', function () {
        infoModalEl.style.display = 'none';
    });
}

if (infoModalEl) {
    // Close modal when clicking outside
    infoModalEl.addEventListener('click', function (e) {
        if (e.target === this) {
            this.style.display = 'none';
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && infoModalEl.style.display === 'flex') {
            infoModalEl.style.display = 'none';
        }
    });
}

// The select shows only the model names; what each model is for goes in the note under it
document.querySelectorAll('select.model-select').forEach(select => {
    const note = document.getElementById(select.getAttribute('aria-describedby').split(' ')[0]);
    const show = () => { note.textContent = select.selectedOptions[0]?.dataset.desc || ''; };
    select.addEventListener('change', show);
    show();
});

// Custom Model Selection Handler - Processing Tab
document.getElementById('process-model-select').addEventListener('change', function () {
    const customModelUpload = document.getElementById('custom-model-upload');
    const modelStatus = document.getElementById('model-status-text');

    if (this.value === 'custom') {
        customModelUpload.style.display = 'block';
        modelStatus.textContent = 'Results go to a folder named after the model file.';
    } else {
        customModelUpload.style.display = 'none';
        modelStatus.textContent = 'Downloaded automatically the first time.';
    }
});

// Custom Model Selection Handler - Diagnostics Tab
document.getElementById('diag-model-select').addEventListener('change', function () {
    const customModelUpload = document.getElementById('diag-custom-model-upload');

    if (this.value === 'custom') {
        customModelUpload.style.display = 'block';
    } else {
        customModelUpload.style.display = 'none';
    }
});


// ==========================================
// Auto-Shutdown Heartbeat & Beacon System
// ==========================================
(function initAutoShutdownBeacon() {
    const tabSessionId = 'tab_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
    const HEARTBEAT_INTERVAL_MS = 2500;

    function sendHeartbeat() {
        fetch('/api/heartbeat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tab_id: tabSessionId }),
            keepalive: true
        }).catch(() => {});
    }

    // Ping iniziale immediato
    sendHeartbeat();

    // Ping periodico
    const intervalId = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);

    // Re-ping al ritorno del focus sulla scheda
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') sendHeartbeat();
    });
    window.addEventListener('focus', sendHeartbeat);

    // 1. Finestra di conferma alla chiusura della scheda o del browser
    window.addEventListener('beforeunload', (e) => {
        e.preventDefault();
        e.returnValue = '';
        return '';
    });

    // 2. Invio del beacon SOLO quando l'utente ha effettivamente confermato l'uscita
    let beaconSent = false;
    function sendShutdownBeacon() {
        if (beaconSent) return;
        beaconSent = true;
        clearInterval(intervalId);
        const payload = JSON.stringify({ tab_id: tabSessionId });

        if (navigator.sendBeacon) {
            const blob = new Blob([payload], { type: 'application/json' });
            navigator.sendBeacon('/api/beacon_shutdown', blob);
        } else {
            fetch('/api/beacon_shutdown', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
                keepalive: true
            }).catch(() => {});
        }
    }

    window.addEventListener('pagehide', sendShutdownBeacon);
    window.addEventListener('unload', sendShutdownBeacon);
})();

// ---------------------------------------------------------------------------
// Cloud Inking (beta): generalist image models through OpenRouter.
// The API key only lives in the input field and in the request body of the
// batch start: it is never put in localStorage, cookies or the URL.
// ---------------------------------------------------------------------------
(function () {
    const tab = document.getElementById('cloud-tab');
    if (!tab) return;

    const $ = id => document.getElementById(id);
    const parseData = name => { try { return JSON.parse(tab.dataset[name] || '{}'); } catch (e) { return {}; } };
    const EST_USD = parseData('estUsd');
    const EST_SEC = parseData('estSec');
    const MODEL_NOTES = parseData('modelNotes');
    const MAX_VARIANTS = parseInt(tab.dataset.maxVariants || '4', 10);
    const FALLBACK_USD = 0.05;

    const keyInput = $('cloud-api-key');
    const consent = $('cloud-consent');
    const modelSel = $('cloud-model-select');
    const customGroup = $('cloud-custom-model-group');
    const customInput = $('cloud-custom-model');
    const variantsInput = $('cloud-variants');
    const startBtn = $('cloud-start-btn');
    const startHint = $('cloud-start-hint');
    const outputBox = $('cloud-output');
    const gallery = $('cloud-gallery');

    const modal = $('cloud-modal');
    const closeBtn = $('cloud-modal-close');
    const cancelBtn = $('cloud-cancel-btn');
    const openBtn = $('cloud-open-folder-btn');
    const viewBtn = $('cloud-view-btn');

    let fileCount = 0;
    let running = false;
    let lastOutputDir = null;

    const fmtUsd = v => `${(v || 0).toFixed(3)} $`;
    const fmtClock = sec => {
        sec = Math.max(0, Math.round(sec));
        return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;
    };
    const fmtInt = n => (n || 0).toLocaleString('en-US');

    function selectedStyle() {
        const el = document.querySelector('input[name="cloud-style"]:checked');
        return el ? el.value : 'fine';
    }

    function selectedModel() {
        return modelSel.value === 'custom' ? customInput.value.trim() : modelSel.value;
    }

    function variants() {
        const v = parseInt(variantsInput.value, 10);
        return Number.isFinite(v) ? Math.min(Math.max(v, 1), MAX_VARIANTS) : 1;
    }

    // ---- optional style reference ---------------------------------------------
    const REF_SLOTS = { pair: ['pencil', 'ink'], sheet: ['sheet'], style: ['style'] };
    const refSlotsBox = $('cloud-ref-slots');
    const refHint = $('cloud-ref-hint');
    const refSection = $('cloud-ref-section');

    function refMode() {
        const el = document.querySelector('input[name="cloud-ref-mode"]:checked');
        return el ? el.value : 'none';
    }

    function refInput(slot) { return $('cloud-ref-' + slot); }

    function refComplete() {
        const mode = refMode();
        if (mode === 'none') return true;
        return REF_SLOTS[mode].every(slot => refInput(slot).files.length > 0);
    }

    function refreshRef() {
        const mode = refMode();
        refSlotsBox.style.display = mode === 'none' ? 'none' : 'grid';
        refSlotsBox.querySelectorAll('.cloud-ref-slot').forEach(el => {
            el.style.display = el.dataset.mode === mode ? '' : 'none';
        });
        refHint.textContent = {
            pair: 'Use a pencil drawing and the inked version of that same drawing, ideally in the style you want for the whole batch.',
            sheet: 'One image with the pencil drawing on the left and its inked version on the right.',
            style: 'One finished ink drawing, with no pencil version: the model copies its line weights and stippling, never its shapes.',
        }[mode] || '';
        refSection.classList.toggle('has-reference', mode !== 'none');
        refreshGate();
    }

    document.querySelectorAll('input[name="cloud-ref-mode"]').forEach(el => el.addEventListener('change', refreshRef));
    document.querySelectorAll('.cloud-ref-slot input[type="file"]').forEach(input => {
        input.addEventListener('change', () => {
            const slot = input.parentElement;
            const thumb = slot.querySelector('.cloud-ref-thumb');
            if (thumb.dataset.url) URL.revokeObjectURL(thumb.dataset.url);
            delete thumb.dataset.url;
            const file = input.files[0];
            if (file && !/\.tiff?$/i.test(file.name)) {
                thumb.dataset.url = URL.createObjectURL(file);
                thumb.src = thumb.dataset.url;
                thumb.alt = file.name;
                thumb.hidden = false;
            } else {
                thumb.hidden = true;
            }
            slot.querySelector('.cloud-ref-remove').hidden = !file;
            refreshGate();
        });
    });
    // Remove: empty the slot so another image can be chosen (clearing the input fires no change event)
    document.querySelectorAll('.cloud-ref-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            const input = btn.parentElement.querySelector('input[type="file"]');
            input.value = '';
            input.dispatchEvent(new Event('change'));
            input.focus();
        });
    });

    function modelIdValid() {
        return modelSel.value !== 'custom' || /^[\w.\-]+\/[\w.\-:]+$/.test(customInput.value.trim());
    }

    // ---- estimate and start gate -------------------------------------------------
    function refreshEstimate() {
        const valueEl = $('cloud-estimate-value');
        const labelEl = $('cloud-estimate-label');
        if (fileCount === 0) {
            valueEl.textContent = '—';
            labelEl.textContent = 'Select drawings to see an estimate';
            return;
        }
        const each = EST_USD[selectedModel()] ?? FALLBACK_USD;
        const known = EST_USD[selectedModel()] !== undefined;
        const v = variants();
        const total = each * fileCount * v;
        valueEl.textContent = `≈ ${total.toFixed(2)} $`;
        labelEl.textContent = `${fileCount} drawing${fileCount > 1 ? 's' : ''} × ${v} version${v > 1 ? 's' : ''} × ~${each.toFixed(2)} $` +
            (known ? '' : ' (price of this model unknown)');
    }

    function refreshGate() {
        const missing = [];
        if (!keyInput.value.trim()) missing.push('your API key');
        if (!consent.checked) missing.push('the privacy notice');
        if (!modelIdValid()) missing.push('a valid model id');
        if (!refComplete()) missing.push({ pair: 'both reference images', sheet: 'the reference sheet', style: 'the reference drawing' }[refMode()]);
        if (fileCount === 0) missing.push('drawings selected from the project');
        startBtn.disabled = running || missing.length > 0;
        startBtn.querySelector('.btn-run-label').textContent = fileCount
            ? `Ink ${fileCount} drawing${fileCount === 1 ? '' : 's'} in the cloud` : 'Ink in the cloud';
        // the connection block folds away once it is complete
        $('cloud-connection').classList.toggle('is-ready', !!keyInput.value.trim() && consent.checked);
        startHint.textContent = missing.length
            ? `To continue: ${missing.join(', ')}.`
            : 'Each image is sent to the provider as soon as you press Start.';
    }

    function refreshAll() {
        refreshEstimate();
        refreshGate();
    }

    document.addEventListener('picker-change', e => {
        if (e.detail.mode !== 'cloud') return;
        fileCount = e.detail.count || 0;
        refreshAll();
    });
    [keyInput, consent, customInput].forEach(el => el.addEventListener('input', refreshGate));
    consent.addEventListener('change', refreshGate);
    variantsInput.addEventListener('input', refreshEstimate);
    variantsInput.addEventListener('change', () => { variantsInput.value = variants(); refreshEstimate(); });
    document.querySelectorAll('input[name="cloud-style"]').forEach(el => el.addEventListener('change', refreshEstimate));

    modelSel.addEventListener('change', () => {
        const custom = modelSel.value === 'custom';
        customGroup.style.display = custom ? 'block' : 'none';
        $('cloud-model-note').textContent = custom ? 'Not tested for this task.' : (MODEL_NOTES[modelSel.value] || '');
        refreshAll();
    });
    customInput.addEventListener('input', refreshEstimate);

    // ---- run modal -------------------------------------------------------------
    let timer = null;
    let startedAt = 0;
    let lastSnap = null;
    let lastSnapAt = 0;
    let sessionId = null;
    let source = null;
    let finished = false;

    function openModal() {
        finished = false;
        $('cloud-run-status').textContent = 'Starting...';
        $('cloud-run-current').textContent = '';
        setProgress($('cloud-run-fill'), 0);
        $('cloud-run-bar').setAttribute('aria-valuenow', '0');
        $('cloud-stat-done').textContent = '0 / 0';
        $('cloud-stat-cost').textContent = fmtUsd(0);
        $('cloud-stat-est').textContent = 'estimate —';
        $('cloud-stat-tokens-box').style.display = 'none';
        $('cloud-stat-time').textContent = '0:00';
        $('cloud-stat-eta').textContent = '';
        const msg = $('cloud-run-message');
        msg.style.display = 'none';
        msg.className = 'cloud-run-message';
        cancelBtn.style.display = '';
        cancelBtn.disabled = false;
        openBtn.style.display = 'none';
        viewBtn.style.display = 'none';
        closeBtn.style.display = 'none';
        modal.style.display = 'flex';
        startedAt = Date.now();
        lastSnap = null;
        clearInterval(timer);
        timer = setInterval(tick, 1000);
        tick();
    }

    function closeModal() {
        if (!finished) return;
        modal.style.display = 'none';
    }

    function tick() {
        const now = Date.now();
        $('cloud-stat-time').textContent = fmtClock((now - startedAt) / 1000);
        if (!finished && lastSnap && lastSnap.eta != null && lastSnap.done < lastSnap.total) {
            const left = lastSnap.eta - (now - lastSnapAt) / 1000;
            $('cloud-stat-eta').textContent = left > 1 ? `about ${fmtClock(left)} left` : 'almost done';
        }
    }

    function applySnapshot(s) {
        lastSnap = s;
        lastSnapAt = Date.now();
        const pct = s.total ? (s.done / s.total) * 100 : 0;
        setProgress($('cloud-run-fill'), pct);
        $('cloud-run-bar').setAttribute('aria-valuenow', String(Math.round(pct)));
        $('cloud-stat-done').textContent = `${s.done} / ${s.total}` + (s.failed ? ` (${s.failed} failed)` : '');
        $('cloud-stat-cost').textContent = fmtUsd(s.cost_usd);
        $('cloud-stat-est').textContent = `estimate ${fmtUsd(s.estimated_usd)}`;
        if (s.has_tokens) {
            $('cloud-stat-tokens-box').style.display = '';
            $('cloud-stat-tokens').textContent = fmtInt(s.tokens_in + s.tokens_out);
            $('cloud-stat-tokens-sub').textContent = `${fmtInt(s.tokens_in)} in · ${fmtInt(s.tokens_out)} out`;
        }
        if (!finished && !cancelBtn.disabled) {
            $('cloud-run-status').textContent = 'Inking your drawings...';
        }
        $('cloud-run-current').textContent = s.current || '';
        tick();
    }

    function showRunMessage(kind, text) {
        const msg = $('cloud-run-message');
        msg.className = `cloud-run-message is-${kind}`;
        msg.textContent = text;
        msg.style.display = 'block';
    }

    const STOP_TEXT = {
        cancelled: ['Stopped', 'warning', 'You stopped the run. Images already finished are saved.'],
        budget: ['Stopped at your limit', 'warning', 'The cost limit was reached, so no new images were started. Finished images are saved.'],
        credits: ['Out of credits', 'error', 'Your OpenRouter account has no credits left. Add credits at openrouter.ai and run the remaining drawings again.'],
        auth: ['API key rejected', 'error', 'OpenRouter did not accept the key. Check that it is correct and active, then try again.'],
        ssl: ['Connection blocked', 'error', 'The secure connection to OpenRouter was blocked, usually by an antivirus that scans HTTPS. Install the optional package truststore (pip install truststore) in the PyPottery environment, restart the app and try again.']
    };

    function finishRun(results, fatalMessage) {
        finished = true;
        running = false;
        clearInterval(timer);
        if (source) { source.close(); source = null; }
        cancelBtn.style.display = 'none';
        closeBtn.style.display = '';
        refreshGate();
        if (results) PyProjects.refresh();   // the run is now in the project history

        if (!results) {
            $('cloud-run-status').textContent = 'Something went wrong';
            showRunMessage('error', fatalMessage || 'The run stopped unexpectedly.');
            return;
        }

        lastOutputDir = results.output_dir;
        const stop = results.stopped && STOP_TEXT[results.stopped];
        if (stop) {
            $('cloud-run-status').textContent = stop[0];
            showRunMessage(stop[1], stop[2]);
        } else if (results.failed > 0) {
            $('cloud-run-status').textContent = `Done, ${results.failed} failed`;
            const first = results.errors && results.errors[0];
            showRunMessage('warning', first ? `${first.file}: ${first.message}` : 'Some images failed. See the log file.');
        } else {
            $('cloud-run-status').textContent = 'Done';
            showRunMessage('success', `${results.successful} image${results.successful === 1 ? '' : 's'} saved. Total cost ${fmtUsd(results.cost_usd)}.`);
        }
        if (results.external) {
            $('cloud-run-message').textContent += results.external.error
                ? ` The copy in your external folder failed (${results.external.error}); the results are complete in the project.`
                : ` ${results.external.copied} copied to your external folder.`;
        }
        $('cloud-run-current').textContent = '';
        $('cloud-stat-cost').textContent = fmtUsd(results.cost_usd);
        $('cloud-stat-done').textContent = `${results.successful} / ${results.total}` + (results.failed ? ` (${results.failed} failed)` : '');
        $('cloud-stat-eta').textContent = '';
        openBtn.style.display = '';
        if (results.successful > 0) viewBtn.style.display = '';
        renderResults(results);
    }

    // ---- results ---------------------------------------------------------------
    function renderResults(results) {
        const box = outputBox;
        box.replaceChildren();
        const p = (html) => { const el = document.createElement('p'); el.innerHTML = html; box.appendChild(el); };
        const msg = document.createElement('div');
        msg.className = 'message message-success';
        msg.innerHTML = '<i class="bi bi-check-circle-fill"></i> <span></span>';
        msg.querySelector('span').textContent = `${results.successful} of ${results.total} images saved.`;
        box.appendChild(msg);
        p(`<strong><i class="bi bi-coin"></i> Cost:</strong> ${fmtUsd(results.cost_usd)}` +
          (results.has_tokens ? ` &middot; ${fmtInt(results.tokens_in)} input and ${fmtInt(results.tokens_out)} output tokens` : ''));
        p(`<strong><i class="bi bi-stopwatch"></i> Average time:</strong> ${results.average_time.toFixed(1)} s per image`);
        (results.folders || []).forEach(folder => {
            const dir = document.createElement('p');
            dir.innerHTML = '<strong><i class="bi bi-folder2-open"></i> Folder:</strong> <code></code>';
            dir.querySelector('code').textContent = `outputs/${folder}`;
            box.appendChild(dir);
        });
        if (results.external) {
            const ext = document.createElement('p');
            ext.textContent = results.external.error
                ? `The copy in your external folder failed (${results.external.error}); the results are complete in the project.`
                : `${results.external.copied} files also copied to ${results.external.dir}.`;
            box.appendChild(ext);
        }
        const folderBtn = document.createElement('button');
        folderBtn.type = 'button';
        folderBtn.className = 'btn btn-open-folder';
        folderBtn.innerHTML = '<i class="bi bi-folder2-open"></i> Open folder';
        folderBtn.addEventListener('click', () => openFolder(lastOutputDir));
        box.appendChild(folderBtn);
        box.style.display = 'block';

        // gallery: the pencil drawing next to each of its inked versions
        const items = results.items || [];
        gallery.replaceChildren();
        if (!items.length) { gallery.style.display = 'none'; return; }
        const stamp = Date.now();
        const title = document.createElement('h3');
        title.textContent = 'Results';
        gallery.appendChild(title);
        const list = document.createElement('div');
        list.className = 'cloud-gallery-list';
        const bySource = new Map();
        items.forEach(it => { if (!bySource.has(it.source)) bySource.set(it.source, []); bySource.get(it.source).push(it); });
        bySource.forEach((versions, source) => {
            const wrap = document.createElement('div');
            const name = document.createElement('p');
            name.className = 'cloud-item-name';
            name.textContent = source;
            const row = document.createElement('div');
            row.className = 'cloud-item-row';
            row.appendChild(figure(`${PyProjects.fileUrl(versions[0].input_path.split('/')[0], versions[0].input_path.split('/').slice(1).join('/'))}?t=${stamp}`,
                versions[0].input_path.startsWith('preprocessed/') ? 'Pencil (preprocessed)' : 'Pencil', source));
            versions.forEach(v => {
                const cap = (versions.length > 1 ? `Version ${v.variant}` : 'Ink') +
                    (v.seconds ? ` · ${v.seconds} s` : '') + (v.cost_usd !== '' && v.cost_usd != null ? ` · ${Number(v.cost_usd).toFixed(3)} $` : '');
                row.appendChild(figure(`${PyProjects.fileUrl('outputs', v.output_path)}?t=${stamp}`, cap, v.file));
            });
            wrap.append(name, row);
            list.appendChild(wrap);
        });
        gallery.appendChild(list);
        gallery.style.display = 'block';
    }

    function figure(src, caption, alt) {
        const fig = document.createElement('figure');
        fig.className = 'cloud-fig';
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.decoding = 'async';
        img.src = src;
        img.alt = alt;
        img.addEventListener('click', () => openLightbox(img.src));
        const cap = document.createElement('figcaption');
        cap.textContent = caption;
        fig.append(img, cap);
        return fig;
    }

    // ---- start / cancel ------------------------------------------------------
    startBtn.addEventListener('click', async function () {
        if (running) return;
        const drawings = await PyProjects.confirmRerun('cloud', PyProjects.selected('cloud'));
        if (!drawings || drawings.length === 0) return;
        running = true;
        refreshGate();
        outputBox.style.display = 'none';
        gallery.style.display = 'none';
        openModal();
        $('cloud-run-status').textContent = 'Preparing...';

        try {
            const reference = refMode();
            if (reference !== 'none') {
                $('cloud-run-status').textContent = 'Uploading the reference...';
                const refForm = new FormData();
                refForm.append('mode', reference);
                REF_SLOTS[reference].forEach(slot => refForm.append(slot, refInput(slot).files[0]));
                const refRes = await fetch('/api/cloud/upload-reference', { method: 'POST', body: refForm });
                const refData = await refRes.json();
                if (!refData.success) throw new Error(refData.error || 'Reference upload failed');
            }

            const response = await fetch('/api/cloud/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_key: keyInput.value.trim(),
                    consent: consent.checked,
                    project_id: PyProjects.currentId(),
                    drawings,
                    use_preprocessed: $('cloud-use-pre').checked,
                    reference_hash: await PyProjects.referenceHash(),
                    model: modelSel.value,
                    custom_model: customInput.value.trim(),
                    style: selectedStyle(),
                    reference: reference,
                    variants: variants(),
                    extra_prompt: $('cloud-extra-prompt').value,
                    align: $('cloud-align').checked,
                    budget_usd: $('cloud-budget').value
                })
            });
            const data = await response.json();
            if (!data.success) throw new Error(data.error || 'Could not start');

            sessionId = data.session_id;
            $('cloud-stat-est').textContent = `estimate ${fmtUsd(data.estimated_usd)}`;
            $('cloud-run-status').textContent = 'Inking your drawings...';
            source = new EventSource(`/api/progress/${sessionId}`);
            source.onmessage = function (event) {
                const msg = JSON.parse(event.data);
                if (msg.keepalive) return;
                if (msg.error) { finishRun(null, msg.message || msg.error); return; }
                if (msg.cloud) applySnapshot(msg.cloud);
                if (msg.completed && msg.results) finishRun(msg.results);
            };
            source.onerror = function () {
                if (finished) return;
                finishRun(null, 'The connection to the app was lost. Images already finished are saved in the output folder.');
            };
        } catch (error) {
            finishRun(null, error.message);
        }
    });

    cancelBtn.addEventListener('click', async function () {
        if (!sessionId || finished) return;
        cancelBtn.disabled = true;
        $('cloud-run-status').textContent = 'Stopping...';
        $('cloud-run-current').textContent = 'Waiting for the images already in progress';
        try { await fetch(`/api/cloud/cancel/${sessionId}`, { method: 'POST' }); } catch (e) { /* the run ends on its own */ }
    });

    openBtn.addEventListener('click', () => { if (lastOutputDir) openFolder(lastOutputDir); });
    viewBtn.addEventListener('click', () => {
        closeModal();
        gallery.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    closeBtn.addEventListener('click', closeModal);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && modal.style.display === 'flex') closeModal(); });

    refreshRef();
    refreshAll();
})();
