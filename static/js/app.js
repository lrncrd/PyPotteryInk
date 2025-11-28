// PyPotteryInk Flask App - Frontend JavaScript

// Splash Screen Management
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
            progressBar.style.width = progress + '%';
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
            reportContainer.innerHTML = `<pre>${data.report}</pre>`;
            reportContainer.style.display = 'block';
        } else {
            showMessage('error', 'Hardware check failed: ' + data.error, reportContainer);
        }
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, reportContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>🔍</span> Check Hardware';
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

    const btn = this;
    const outputContainer = document.getElementById('diagnostics-output');
    const gallery = document.getElementById('diagnostics-gallery');

    btn.disabled = true;
    btn.innerHTML = '<span class="loading"></span> Running...';
    outputContainer.style.display = 'none';
    gallery.style.display = 'none';

    try {
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

        // Run diagnostics
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

        const data = await diagResponse.json();

        if (data.success) {
            showMessage('success', `Diagnostics completed! Generated ${data.diagnostic_files.length} visualizations.`, outputContainer);

            // Display images
            gallery.innerHTML = '';
            data.diagnostic_files.forEach(file => {
                const img = document.createElement('img');
                img.src = `/api/get-image/diagnostics/${file}`;
                img.alt = file;
                gallery.appendChild(img);
            });
            gallery.style.display = 'grid';
        } else {
            showMessage('error', 'Diagnostics failed: ' + data.error, outputContainer);
        }
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>🚀</span> Run Diagnostics';
    }
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
            let html = '<div class="message message-success">✅ Statistics calculated successfully!</div>';
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
        btn.innerHTML = '<span>📊</span> Calculate Statistics';
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

        // Preprocess images
        const preprocessResponse = await fetch('/api/preprocess-images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                calculate_stats: useCalculatedStats,
                stats_file: statsFilePath,
                output_dir: outputDir
            })
        });

        const data = await preprocessResponse.json();

        if (data.success) {
            const html = `
                <div class="message message-success">✅ Preprocessing completed!</div>
                <p><strong>Total processed:</strong> ${data.processed}</p>
                <p><strong>Images adjusted:</strong> ${data.adjusted}</p>
                <p><strong>No adjustments needed:</strong> ${data.processed - data.adjusted}</p>
                <p><strong>Output directory:</strong> ${data.output_dir}</p>
            `;
            outputContainer.innerHTML = html;
            outputContainer.style.display = 'block';
        } else {
            showMessage('error', 'Preprocessing failed: ' + data.error, outputContainer);
        }
    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>✨</span> Apply Preprocessing';
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
        progressFill.style.width = '5%';
        patchProgressFill.style.width = '0%';

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
            progressFill.style.width = '8%';

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
        progressFill.style.width = '10%';

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

        // Start processing
        statusText.textContent = 'Starting processing...';
        progressFill.style.width = '15%';

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

            if (data.error) {
                showMessage('error', data.message, outputContainer);
                eventSource.close();
                btn.disabled = false;
                btn.innerHTML = '<span>🚀</span> Start Processing';
                return;
            }

            // Update overall progress bar and status
            if (data.progress !== undefined) {
                const progress = Math.min(Math.max(data.progress, 0), 100);
                progressFill.style.width = progress + '%';
                statusText.textContent = data.message || 'Processing...';
            }

            // Update patch progress bar and status
            if (data.patch_progress !== undefined) {
                const patchProgress = Math.min(Math.max(data.patch_progress, 0), 100);
                patchProgressFill.style.width = patchProgress + '%';
                patchStatusText.textContent = data.patch_message || 'Processing patches...';
            }

            // If completed
            if (data.completed && data.results) {
                eventSource.close();

                const results = data.results;
                let html = '<div class="message message-success">🎉 Processing completed successfully!</div>';
                html += '<h3>Results Summary:</h3>';
                html += `<p><strong>✅ Successful:</strong> ${results.successful} images</p>`;
                html += `<p><strong>❌ Failed:</strong> ${results.failed} images</p>`;
                html += `<p><strong>⏱️ Average time:</strong> ${results.average_time.toFixed(2)}s per image</p>`;
                html += `<p><strong>📁 Output directory:</strong> <code>${results.output_dir}</code></p>`;

                if (results.log_file) {
                    html += `<p><strong>📝 Log file:</strong> <code>${results.log_file}</code></p>`;
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
                btn.innerHTML = '<span>🚀</span> Start Processing';
            }
        };

        eventSource.onerror = function () {
            eventSource.close();
            showMessage('error', 'Connection lost. Processing may still continue in background.', outputContainer);
            btn.disabled = false;
            btn.innerHTML = '<span>🚀</span> Start Processing';
        };

    } catch (error) {
        showMessage('error', 'Error: ' + error.message, outputContainer);
        btn.disabled = false;
        btn.innerHTML = '<span>🚀</span> Start Processing';
        progressContainer.style.display = 'none';
    }
});

// Helper function to show messages
function showMessage(type, message, container) {
    const messageClass = `message message-${type}`;
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
    container.innerHTML = `<div class="${messageClass}">${icon} ${message}</div>`;
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
                        <div class="tiff-icon">🖼️</div>
                        <div class="tiff-label">TIFF</div>
                    </div>
                    <div class="preview-item-name">${file.name}</div>
                    <button class="preview-item-remove" data-index="${index}">×</button>
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
                        <button class="preview-item-remove" data-index="${index}">×</button>
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

// Dark Mode Toggle
document.getElementById('theme-toggle').addEventListener('click', function () {
    document.body.classList.toggle('dark-mode');
    const icon = this.querySelector('.theme-icon');

    if (document.body.classList.contains('dark-mode')) {
        icon.textContent = '☀️';
        localStorage.setItem('darkMode', 'enabled');
    } else {
        icon.textContent = '🌙';
        localStorage.setItem('darkMode', 'disabled');
    }
});

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'enabled') {
    document.body.classList.add('dark-mode');
    document.querySelector('.theme-icon').textContent = '☀️';
}

// Info Modal
document.getElementById('info-btn').addEventListener('click', function () {
    document.getElementById('info-modal').style.display = 'flex';
});

document.getElementById('close-info-modal').addEventListener('click', function () {
    document.getElementById('info-modal').style.display = 'none';
});

// Close modal when clicking outside
document.getElementById('info-modal').addEventListener('click', function (e) {
    if (e.target === this) {
        this.style.display = 'none';
    }
});

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
        btn.innerHTML = '📁 Browse...';
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
        btn.innerHTML = '📁 Browse...';
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
        btn.innerHTML = '📁 Browse...';
    }
});
