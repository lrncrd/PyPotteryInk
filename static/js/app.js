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

// File Upload Handler
function setupFileUpload(inputId, enableButtonIds) {
    const input = document.getElementById(inputId);
    input.addEventListener('change', function () {
        const hasFiles = this.files.length > 0;
        enableButtonIds.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) btn.disabled = !hasFiles;
        });
    });
}

// Setup file uploads for different tabs
setupFileUpload('diag-image-upload', ['run-diagnostics-btn']);
setupFileUpload('stats-image-upload', ['calculate-stats-btn']);
setupFileUpload('preprocess-image-upload', ['preprocess-btn']);

// Process image upload with counter
document.getElementById('process-image-upload').addEventListener('change', function () {
    const hasFiles = this.files.length > 0;
    const uploadCount = document.getElementById('upload-count');
    const processBtn = document.getElementById('process-images-btn');

    if (hasFiles) {
        uploadCount.textContent = `${this.files.length} file(s) selected`;
        processBtn.disabled = false;
    } else {
        uploadCount.textContent = '';
        processBtn.disabled = true;
    }
});

// Diagnostics
document.getElementById('run-diagnostics-btn').addEventListener('click', async function () {
    const files = document.getElementById('diag-image-upload').files;
    const modelName = document.getElementById('diag-model-select').value;
    const patchSize = document.getElementById('diag-patch-size').value;
    const overlap = document.getElementById('diag-overlap').value;
    const contrastValues = document.getElementById('diag-contrast').value;

    // Max 5 images check
    if (files.length > 5) {
        alert('Maximum 5 images allowed for diagnostics. Please select fewer images.');
        return;
    }

    const btn = this;
    const outputContainer = document.getElementById('diagnostics-output');
    const gallery = document.getElementById('diagnostics-gallery');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Running...';
    outputContainer.style.display = 'none';
    gallery.style.display = 'none';
    gallery.innerHTML = '';

    try {
        // Clear diagnostics folder first
        await fetch('/api/clear-diagnostics', { method: 'POST' });

        // Upload images
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        const uploadResponse = await fetch('/api/upload-images', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');

        // Handle custom model upload if selected
        if (modelName === 'custom') {
            const customModelFile = document.getElementById('diag-custom-model-file').files[0];
            if (!customModelFile) {
                throw new Error('Please select a custom model file');
            }

            const modelFormData = new FormData();
            modelFormData.append('model', customModelFile);

            const modelUploadResponse = await fetch('/api/upload-custom-model', {
                method: 'POST',
                body: modelFormData
            });

            if (!modelUploadResponse.ok) {
                const error = await modelUploadResponse.json();
                throw new Error(error.error || 'Custom model upload failed');
            }
        }

        // Check if diffusion model (sd-turbo) is cached
        const diffusionCheckResponse = await fetch('/api/check-diffusion-model');
        const diffusionCheck = await diffusionCheckResponse.json();

        // If sd-turbo is not cached, show download overlay
        if (!diffusionCheck.cached) {
            showDownloadOverlay('Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).');
        }

        // Start diagnostics (server will run in background and return a session_id)
        const diagResponse = await fetch('/api/diagnostics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                patch_size: parseInt(patchSize),
                overlap: parseInt(overlap),
                contrast_values: contrastValues
            })
        });

        if (!diagResponse.ok) throw new Error('Failed to start diagnostics');

        const resp = await diagResponse.json();
        console.log('Diagnostics start response:', resp);

        // Backwards-compatible: if server returned results directly
        if (resp.success && resp.diagnostic_files) {
            showDiagnosticsComplete(resp.diagnostic_files, outputContainer, gallery);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Run Diagnostics';
            return;
        }

        if (!resp.success || !resp.session_id) {
            throw new Error(resp.error || 'Failed to start diagnostics');
        }

        const sessionId = resp.session_id;
        const progressContainer = document.getElementById('diag-progress');
        const progressFill = document.getElementById('diag-progress-fill');
        const statusText = document.getElementById('diag-status');

        progressContainer.style.display = 'block';
        setProgress(progressFill, 5);
        statusText.textContent = 'Starting diagnostics...';

        const eventSource = new EventSource(`/api/progress/${sessionId}`);

        eventSource.onmessage = function (event) {
            const data = JSON.parse(event.data);
            if (data.keepalive) return;

            // Model-download progress (style .pkl or the shared sd-turbo
            // backbone) drives the overlay's real progress bar instead of
            // the main diagnostics progress bar, and doesn't mean
            // diagnostics has actually started yet.
            if (data.type === 'style_model_download' || data.type === 'sdturbo_download') {
                showDownloadOverlay(data.type === 'sdturbo_download'
                    ? 'Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).'
                    : 'Downloading style model...');
                updateDownloadProgress(data.progress, data.message);
                return;
            }

            // Any other message means real diagnostics progress has begun -
            // hide the download overlay if it was showing.
            hideDownloadOverlay();

            if (data.error) {
                showMessage('error', data.message || 'Diagnostics error', outputContainer);
                eventSource.close();
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Run Diagnostics';
                return;
            }

            if (data.progress !== undefined) {
                const progress = Math.min(Math.max(data.progress, 0), 100);
                setProgress(progressFill, progress);
                statusText.textContent = data.message || 'Running diagnostics...';
            }

            if (data.completed && data.results) {
                eventSource.close();

                const files = data.results.diagnostic_files || [];
                showDiagnosticsComplete(files, outputContainer, gallery);

                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Run Diagnostics';
                setProgress(progressFill, 100);
                statusText.textContent = 'Completed';
            }
        };

        eventSource.onerror = function () {
            eventSource.close();
            showMessage('error', 'Connection lost. Diagnostics may still continue in background.', outputContainer);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Run Diagnostics';
            progressContainer.style.display = 'none';
        };
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Run Diagnostics';
    }
});

// Helper function to show diagnostics complete with open folder button
function showDiagnosticsComplete(files, outputContainer, gallery) {
    let html = `<div class="message message-success"><i class="bi bi-check-circle-fill"></i> Diagnostics completed! Generated ${files.length} visualizations.</div>`;
    html += `<button class="btn btn-open-folder" onclick="openDiagnosticsFolder()"><i class="bi bi-folder2-open"></i> Open Diagnostics Folder</button>`;
    outputContainer.innerHTML = html;
    outputContainer.style.display = 'block';

    gallery.innerHTML = '';
    files.forEach(file => {
        const img = document.createElement('img');
        img.src = `/api/get-image/diagnostics/${file}`;
        img.alt = file;
        img.onclick = function () { openLightbox(this.src); };
        gallery.appendChild(img);
    });
    gallery.style.display = 'grid';
}

// Open diagnostics folder
async function openDiagnosticsFolder() {
    try {
        await fetch('/api/open-folder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: 'temp_diagnostics' })
        });
    } catch (error) {
        console.error('Failed to open folder:', error);
    }
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

// Calculate Statistics
document.getElementById('calculate-stats-btn').addEventListener('click', async function () {
    const files = document.getElementById('stats-image-upload').files;
    const savePath = document.getElementById('stats-save-path').value;

    const btn = this;
    const outputContainer = document.getElementById('stats-output');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Calculating...';
    outputContainer.style.display = 'none';

    try {
        // Upload images
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        const uploadResponse = await fetch('/api/upload-images', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');

        // Calculate statistics
        const statsResponse = await fetch('/api/calculate-statistics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ save_path: savePath })
        });

        const data = await statsResponse.json();

        if (data.success) {
            let html = '<div class="message message-success"><i class="bi bi-check-circle-fill"></i> Statistics calculated successfully!</div>';
            html += `<p><strong>Images analyzed:</strong> ${data.summary.images_analyzed}</p>`;
            html += `<p><strong>Statistics file:</strong> ${data.summary.statistics_file}</p>`;
            html += '<h4>Distributions:</h4><table style="width: 100%; border-collapse: collapse;">';
            html += '<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>';

            for (const [metric, stats] of Object.entries(data.summary.distributions)) {
                html += `<tr>
                    <td>${metric.replace(/_/g, ' ')}</td>
                    <td>${stats.mean.toFixed(4)}</td>
                    <td>${stats.std.toFixed(4)}</td>
                    <td>${stats.min.toFixed(4)}</td>
                    <td>${stats.max.toFixed(4)}</td>
                    <td>${stats.median.toFixed(4)}</td>
                </tr>`;
            }
            html += '</table>';

            outputContainer.innerHTML = html;
            outputContainer.style.display = 'block';
        } else {
            showMessage('error', 'Statistics calculation failed: ' + data.error, outputContainer);
        }
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-bar-chart-line"></i> Calculate Statistics';
    }
});

// Preprocessing
document.getElementById('preprocess-btn').addEventListener('click', async function () {
    const files = document.getElementById('preprocess-image-upload').files;
    const useCalculatedStats = document.getElementById('use-calculated-stats').checked;
    const outputDir = document.getElementById('preprocess-output-dir').value;

    const btn = this;
    const outputContainer = document.getElementById('preprocess-output');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Processing...';
    outputContainer.style.display = 'none';

    try {
        // Upload images
        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        const uploadResponse = await fetch('/api/upload-images', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');

        let statsFilePath = null;

        // Handle statistics file if not using calculated stats
        if (!useCalculatedStats) {
            const statsFile = document.getElementById('stats-file-upload').files[0];
            if (!statsFile) {
                throw new Error('Please select a statistics file (.npy) or check "Use calculated statistics"');
            }

            const statsFormData = new FormData();
            statsFormData.append('stats_file', statsFile);

            const statsUploadResponse = await fetch('/api/upload-stats', {
                method: 'POST',
                body: statsFormData
            });

            if (!statsUploadResponse.ok) {
                const error = await statsUploadResponse.json();
                throw new Error(error.error || 'Statistics file upload failed');
            }

            const statsData = await statsUploadResponse.json();
            statsFilePath = statsData.stats_path;
        }

        console.log('Starting preprocess flow: uploading images and requesting preprocessing');
        // Start preprocessing (server runs it in background and returns a session_id)
        const preprocessResponse = await fetch('/api/preprocess-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                calculate_stats: useCalculatedStats,
                stats_file: statsFilePath,
                output_dir: outputDir
            })
        });

        if (!preprocessResponse.ok) throw new Error('Failed to start preprocessing');

        const respData = await preprocessResponse.json();
        console.log('Preprocess start response:', respData);

        // Backwards-compatible: if server returned final results directly, show them
        if (respData.success && respData.processed !== undefined) {
            console.log('Server returned direct results (no SSE).');
            const html = `
                <div class="message message-success"><i class="bi bi-check-circle-fill"></i> Preprocessing completed!</div>
                <p><strong>Total processed:</strong> ${respData.processed}</p>
                <p><strong>Images adjusted:</strong> ${respData.adjusted}</p>
                <p><strong>No adjustments needed:</strong> ${respData.processed - respData.adjusted}</p>
                <p><strong>Output directory:</strong> ${respData.output_dir}</p>
            `;
            outputContainer.innerHTML = html;
            outputContainer.style.display = 'block';
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-stars"></i> Apply Preprocessing';
            return;
        }

        if (!respData.success || !respData.session_id) {
            throw new Error(respData.error || 'Failed to start preprocessing');
        }

        // Show progress UI and listen for SSE
        const sessionId = respData.session_id;
        const progressContainer = document.getElementById('preprocess-progress');
        const progressFill = document.getElementById('preprocess-progress-fill');
        const statusText = document.getElementById('preprocess-status');

        progressContainer.style.display = 'block';
        setProgress(progressFill, 5);
        statusText.textContent = 'Starting preprocessing...';

        const eventSource = new EventSource(`/api/progress/${sessionId}`);

        eventSource.onmessage = function (event) {
            const data = JSON.parse(event.data);

            if (data.keepalive) return;

            if (data.error) {
                showMessage('error', data.message || 'Preprocessing error', outputContainer);
                eventSource.close();
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-stars"></i> Apply Preprocessing';
                return;
            }

            if (data.progress !== undefined) {
                const progress = Math.min(Math.max(data.progress, 0), 100);
                setProgress(progressFill, progress);
                statusText.textContent = data.message || 'Processing...';
            }

            if (data.completed && data.results) {
                eventSource.close();

                const results = data.results;
                const html = `
                    <div class="message message-success"><i class="bi bi-check-circle-fill"></i> Preprocessing completed!</div>
                    <p><strong>Total processed:</strong> ${results.processed}</p>
                    <p><strong>Images adjusted:</strong> ${results.adjusted}</p>
                    <p><strong>No adjustments needed:</strong> ${results.processed - results.adjusted}</p>
                    <p><strong>Output directory:</strong> ${results.output_dir}</p>
                `;

                outputContainer.innerHTML = html;
                outputContainer.style.display = 'block';

                // Reset button
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-stars"></i> Apply Preprocessing';
                setProgress(progressFill, 100);
                statusText.textContent = 'Completed';
            }
        };

        eventSource.onerror = function () {
            eventSource.close();
            showMessage('error', 'Connection lost. Preprocessing may still continue in background.', outputContainer);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-stars"></i> Apply Preprocessing';
            progressContainer.style.display = 'none';
        };
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-stars"></i> Apply Preprocessing';
    }
});

// Batch Processing with Real-time Progress
document.getElementById('process-images-btn').addEventListener('click', async function () {
    const files = document.getElementById('process-image-upload').files;
    const modelName = document.getElementById('process-model-select').value;
    const outputDir = document.getElementById('process-output-dir').value;
    const patchSize = document.getElementById('process-patch-size').value;
    const overlap = document.getElementById('process-overlap').value;
    const contrastScale = document.getElementById('process-contrast').value;
    const upscale = document.getElementById('process-upscale').value;
    const useFp16 = document.getElementById('use-fp16').checked;

    const btn = this;
    const progressContainer = document.getElementById('process-progress');
    const progressFill = document.getElementById('process-progress-fill');
    const patchProgressFill = document.getElementById('patch-progress-fill');
    const statusText = document.getElementById('process-status');
    const patchStatusText = document.getElementById('patch-status');
    const outputContainer = document.getElementById('process-output');
    const gallery = document.getElementById('process-gallery');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Processing...';
    progressContainer.style.display = 'block';
    outputContainer.style.display = 'none';
    gallery.style.display = 'none';

    try {
        // Upload images
        statusText.textContent = 'Uploading images...';
        patchStatusText.textContent = 'Waiting...';
        setProgress(progressFill, 5);
        setProgress(patchProgressFill, 0);

        const formData = new FormData();
        Array.from(files).forEach(file => formData.append('files', file));

        const uploadResponse = await fetch('/api/upload-images', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');

        // Handle custom model upload if selected
        if (modelName === 'custom') {
            const customModelFile = document.getElementById('custom-model-file').files[0];
            if (!customModelFile) {
                throw new Error('Please select a custom model file');
            }

            statusText.textContent = 'Uploading custom model...';
            setProgress(progressFill, 8);

            const modelFormData = new FormData();
            modelFormData.append('model', customModelFile);

            const modelUploadResponse = await fetch('/api/upload-custom-model', {
                method: 'POST',
                body: modelFormData
            });

            if (!modelUploadResponse.ok) {
                const error = await modelUploadResponse.json();
                throw new Error(error.error || 'Custom model upload failed');
            }
        }

        // Check and download model if needed (for non-custom models)
        statusText.textContent = 'Checking model...';
        setProgress(progressFill, 10);

        // Only check model if not custom
        if (modelName !== 'custom') {
            const modelCheckResponse = await fetch('/api/check-model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            });

            const modelCheck = await modelCheckResponse.json();
            if (!modelCheck.success) {
                throw new Error('Model not available: ' + modelCheck.error);
            }
        }

        // Check if diffusion model (sd-turbo) is cached
        statusText.textContent = 'Checking diffusion model...';
        setProgress(progressFill, 12);

        const diffusionCheckResponse = await fetch('/api/check-diffusion-model');
        const diffusionCheck = await diffusionCheckResponse.json();

        // If sd-turbo is not cached, show download overlay and simulate progress
        // The actual download happens when the model is loaded
        if (!diffusionCheck.cached) {
            showDownloadOverlay('Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).');
            // Don't await the simulation - let it run in the background
            // The overlay will be hidden when processing starts successfully
        }

        // Start processing
        statusText.textContent = 'Starting processing...';
        setProgress(progressFill, 15);

        const processResponse = await fetch('/api/process-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_name: modelName,
                output_dir: outputDir,
                use_fp16: useFp16,
                contrast_scale: parseFloat(contrastScale),
                patch_size: parseInt(patchSize),
                overlap: parseInt(overlap),
                upscale: parseFloat(upscale)
            })
        });

        const processData = await processResponse.json();

        if (!processData.success) {
            throw new Error('Failed to start processing: ' + processData.error);
        }

        // Listen to progress updates via Server-Sent Events
        const sessionId = processData.session_id;
        const eventSource = new EventSource(`/api/progress/${sessionId}`);

        eventSource.onmessage = function (event) {
            const data = JSON.parse(event.data);

            if (data.keepalive) {
                // Just a keepalive, ignore
                return;
            }

            // Model-download progress (style .pkl or the shared sd-turbo
            // backbone) drives the overlay's real progress bar instead of
            // the main processing progress bar, and doesn't mean processing
            // has actually started yet.
            if (data.type === 'style_model_download' || data.type === 'sdturbo_download') {
                showDownloadOverlay(data.type === 'sdturbo_download'
                    ? 'Downloading stabilityai/sd-turbo components. This is a one-time download (~5GB).'
                    : 'Downloading style model...');
                updateDownloadProgress(data.progress, data.message);
                return;
            }

            // Any other message means real processing progress has begun -
            // hide the download overlay if it was showing.
            hideDownloadOverlay();

            if (data.error) {
                showMessage('error', data.message, outputContainer);
                eventSource.close();
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Start Processing';
                return;
            }

            // Update overall progress bar and status
            if (data.progress !== undefined) {
                const progress = Math.min(Math.max(data.progress, 0), 100);
                setProgress(progressFill, progress);
                statusText.textContent = data.message || 'Processing...';
            }

            // Update patch progress bar and status
            if (data.patch_progress !== undefined) {
                const patchProgress = Math.min(Math.max(data.patch_progress, 0), 100);
                setProgress(patchProgressFill, patchProgress);
                patchStatusText.textContent = data.patch_message || 'Processing patches...';
            }

            // If completed
            if (data.completed && data.results) {
                eventSource.close();

                const results = data.results;
                let html = '<div class="message message-success"><i class="bi bi-check-circle-fill"></i> Processing completed successfully!</div>';
                html += '<h3>Results Summary:</h3>';
                html += `<p><strong><i class="bi bi-check-circle-fill" style="color: var(--teal);"></i> Successful:</strong> ${results.successful} images</p>`;
                html += `<p><strong><i class="bi bi-x-circle-fill" style="color: var(--danger-color);"></i> Failed:</strong> ${results.failed} images</p>`;
                html += `<p><strong><i class="bi bi-stopwatch"></i> Average time:</strong> ${results.average_time.toFixed(2)}s per image</p>`;
                html += `<p><strong><i class="bi bi-folder2-open"></i> Output directory:</strong> <code>${results.output_dir}</code></p>`;

                if (results.log_file) {
                    html += `<p><strong><i class="bi bi-file-earmark-text"></i> Log file:</strong> <code>${results.log_file}</code></p>`;
                }

                outputContainer.innerHTML = html;
                outputContainer.style.display = 'block';

                // Display comparison images if available
                if (results.comparison_images && results.comparison_images.length > 0) {
                    gallery.innerHTML = '<h3>Comparison Images:</h3>';
                    const galleryDiv = document.createElement('div');
                    galleryDiv.className = 'gallery';

                    results.comparison_images.slice(0, 20).forEach(file => {
                        const img = document.createElement('img');
                        img.src = `/api/get-image/comparisons/${file}`;
                        img.alt = file;
                        img.onclick = function () { openLightbox(this.src); };
                        galleryDiv.appendChild(img);
                    });

                    gallery.appendChild(galleryDiv);
                    gallery.style.display = 'block';

                    if (results.comparison_images.length > 20) {
                        const note = document.createElement('p');
                        note.textContent = `Showing 20 of ${results.comparison_images.length} comparison images`;
                        note.style.color = '#666';
                        gallery.appendChild(note);
                    }
                }

                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Start Processing';
            }
        };

        eventSource.onerror = function () {
            eventSource.close();
            showMessage('error', 'Connection lost. Processing may still continue in background.', outputContainer);
            btn.disabled = false;
            btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Start Processing';
        };

    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Start Processing';
        progressContainer.style.display = 'none';
    }
});

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

// Stats file upload toggle
// Stats file upload toggle - Removed to keep it always visible
// document.getElementById('use-calculated-stats').addEventListener('change', function () {
//     const statsFileGroup = document.getElementById('stats-file-group');
//     statsFileGroup.style.display = this.checked ? 'none' : 'block';
// });

// Initialize image upload area with preview
function initImageUpload(areaId, inputId, previewId) {
    const uploadArea = document.getElementById(areaId);
    const fileInput = document.getElementById(inputId);
    const previewContainer = document.getElementById(previewId);

    if (!uploadArea || !fileInput || !previewContainer) return;

    let selectedFiles = [];

    // Click to upload
    uploadArea.addEventListener('click', function (e) {
        if (!e.target.classList.contains('preview-item-remove')) {
            fileInput.click();
        }
    });

    // File selection
    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function (e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('dragover');

        const files = e.dataTransfer.files;
        handleFiles(files);
    });

    function handleFiles(files) {
        selectedFiles = Array.from(files);

        if (selectedFiles.length === 0) {
            previewContainer.style.display = 'none';
            uploadArea.querySelector('.upload-placeholder').style.display = 'flex';
            return;
        }

        // Update the file input
        const dataTransfer = new DataTransfer();
        selectedFiles.forEach(file => dataTransfer.items.add(file));
        fileInput.files = dataTransfer.files;

        // Show preview
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        previewContainer.style.display = 'grid';
        previewContainer.innerHTML = '';

        selectedFiles.forEach((file, index) => {
            const previewItem = document.createElement('div');
            previewItem.className = 'preview-item';

            // Check if file is a TIFF
            const isTiff = file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff');

            if (isTiff) {
                // Show placeholder for TIFF files
                previewItem.innerHTML = `
                    <div class="tiff-placeholder">
                        <div class="tiff-icon"><i class="bi bi-file-earmark-image"></i></div>
                        <div class="tiff-label">TIFF</div>
                    </div>
                    <div class="preview-item-name">${file.name}</div>
                    <button class="preview-item-remove" data-index="${index}" aria-label="Remove image"><i class="bi bi-x"></i></button>
                `;
                previewContainer.appendChild(previewItem);

                // Remove button
                previewItem.querySelector('.preview-item-remove').addEventListener('click', function (e) {
                    e.stopPropagation();
                    removeFile(parseInt(this.getAttribute('data-index')));
                });
            } else {
                // Show preview for other image formats
                const reader = new FileReader();
                reader.onload = function (e) {
                    previewItem.innerHTML = `
                        <img src="${e.target.result}" alt="${file.name}">
                        <div class="preview-item-name">${file.name}</div>
                        <button class="preview-item-remove" data-index="${index}" aria-label="Remove image"><i class="bi bi-x"></i></button>
                    `;
                    previewContainer.appendChild(previewItem);

                    // Remove button
                    previewItem.querySelector('.preview-item-remove').addEventListener('click', function (e) {
                        e.stopPropagation();
                        removeFile(parseInt(this.getAttribute('data-index')));
                    });
                };
                reader.readAsDataURL(file);
            }
        });

        // Add upload count
        const countDiv = document.createElement('div');
        countDiv.className = 'upload-count';
        countDiv.textContent = `${selectedFiles.length} image${selectedFiles.length > 1 ? 's' : ''} selected`;
        previewContainer.appendChild(countDiv);

        // Enable buttons based on upload area
        updateButtonStates(areaId);
    }

    function removeFile(index) {
        selectedFiles.splice(index, 1);
        handleFiles(selectedFiles);
    }

    function updateButtonStates(areaId) {
        const hasFiles = selectedFiles.length > 0;

        if (areaId === 'stats-upload-area') {
            document.getElementById('calculate-stats-btn').disabled = !hasFiles;
        } else if (areaId === 'preprocess-upload-area') {
            document.getElementById('preprocess-btn').disabled = !hasFiles;
        } else if (areaId === 'diag-upload-area') {
            document.getElementById('run-diagnostics-btn').disabled = !hasFiles;
        } else if (areaId === 'process-upload-area') {
            document.getElementById('process-images-btn').disabled = !hasFiles;
        }
    }
}

// Initialize all upload areas on load
document.addEventListener('DOMContentLoaded', function () {
    initImageUpload('stats-upload-area', 'stats-image-upload', 'stats-preview');
    initImageUpload('preprocess-upload-area', 'preprocess-image-upload', 'preprocess-preview');
    initImageUpload('diag-upload-area', 'diag-image-upload', 'diag-preview');
    initImageUpload('process-upload-area', 'process-image-upload', 'process-preview');
});

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

// Custom Model Selection Handler - Processing Tab
document.getElementById('process-model-select').addEventListener('change', function () {
    const customModelUpload = document.getElementById('custom-model-upload');
    const modelStatus = document.getElementById('model-status-text');

    if (this.value === 'custom') {
        customModelUpload.style.display = 'block';
        modelStatus.textContent = 'Select a custom .pkl model file from your computer';
    } else {
        customModelUpload.style.display = 'none';
        modelStatus.textContent = 'Model will be automatically downloaded if not present';
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

// Directory Browser - Processing Tab
document.getElementById('browse-output-dir').addEventListener('click', async function () {
    const btn = this;
    const input = document.getElementById('process-output-dir');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Opening...';

    try {
        const response = await fetch('/api/select-directory', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success && data.directory) {
            input.value = data.directory;
        } else {
            console.log('No directory selected');
        }
    } catch (error) {
        console.error('Error selecting directory:', error);
        alert('Error opening directory picker: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-folder2-open"></i> Browse...';
    }
});

// Directory Browser - Preprocessing Tab
document.getElementById('browse-preprocess-dir').addEventListener('click', async function () {
    const btn = this;
    const input = document.getElementById('preprocess-output-dir');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Opening...';

    try {
        const response = await fetch('/api/select-directory', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success && data.directory) {
            input.value = data.directory;
        } else {
            console.log('No directory selected');
        }
    } catch (error) {
        console.error('Error selecting directory:', error);
        alert('Error opening directory picker: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-folder2-open"></i> Browse...';
    }
});

// Directory Browser - Statistics Save Path
document.getElementById('browse-stats-path').addEventListener('click', async function () {
    const btn = this;
    const input = document.getElementById('stats-save-path');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Opening...';

    try {
        const response = await fetch('/api/select-directory', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success && data.directory) {
            // Add the default filename to the selected directory
            input.value = data.directory + '/custom_stats.npy';
        } else {
            console.log('No directory selected');
        }
    } catch (error) {
        console.error('Error selecting directory:', error);
        alert('Error opening directory picker: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-folder2-open"></i> Browse...';
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

