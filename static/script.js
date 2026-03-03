document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const changeBtn = document.getElementById('change-btn');
    const fileNameDisplay = document.getElementById('file-name-display');
    const loadingOverlay = document.getElementById('loading-spinner');
    const resultsCard = document.getElementById('results-card');

    // Result elements
    const resultStatus = document.getElementById('result-status');
    const realMeter = document.getElementById('real-meter');
    const fakeMeter = document.getElementById('fake-meter');
    const realText = document.getElementById('real-text');
    const fakeText = document.getElementById('fake-text');
    const resetBtn = document.getElementById('reset-btn');
    const reportBtn = document.getElementById('report-btn');

    let currentFile = null;

    // --- Drag and Drop Handlers ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    // Fallback to click upload if not interacting with preview controls
    dropZone.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
            fileInput.click();
        }
    });

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            // Validate image type
            if (!file.type.match('image.*')) {
                alert("Please upload a valid image file (JPG, PNG).");
                return;
            }
            currentFile = file;
            previewFile(file);
        }
    }

    // --- UI State Management ---

    function previewFile(file) {
        let reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            // Update preview image
            imagePreview.src = reader.result;
            fileNameDisplay.textContent = file.name;

            // Toggle visibility
            uploadPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultsCard.classList.add('hidden'); // Ensure results are hidden if selecting new image
        }
    }

    changeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Stop bubbling to dropZone
        resetUI();
        fileInput.click();
    });

    function resetUI() {
        currentFile = null;
        fileInput.value = "";
        uploadPrompt.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultsCard.classList.add('hidden');
        dropZone.classList.remove('hidden');
        // Reset meters
        realMeter.style.width = '0%';
        fakeMeter.style.width = '0%';
        realText.textContent = '0%';
        fakeText.textContent = '0%';
        resultStatus.className = '';
        reportBtn.classList.add('hidden');
    }

    resetBtn.addEventListener('click', resetUI);

    // --- API Interaction ---

    analyzeBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!currentFile) return;

        // Show loading state
        loadingOverlay.classList.remove('hidden');
        resultsCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            displayResults(data);

        } catch (error) {
            console.error('Error during analysis:', error);
            alert("An error occurred during analysis: " + error.message);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    });

    function displayResults(data) {
        // Hide upload area slightly to emphasize results, or just show results below
        // we'll keep drop zone visible but maybe resize it if needed, or just show results below it.
        resultsCard.classList.remove('hidden');

        // Trigger reflow for animations
        void resultsCard.offsetWidth;

        const isReal = data.status === 'REAL';
        const realPercentage = (data.real_confidence * 100).toFixed(2);
        const fakePercentage = (data.fake_confidence * 100).toFixed(2);

        // Update Text
        resultStatus.textContent = isReal ? "Authentic Image" : "Manipulated (Deepfake)";
        resultStatus.className = isReal ? 'status-real' : 'status-fake';

        realText.textContent = `${realPercentage}%`;
        fakeText.textContent = `${fakePercentage}%`;

        // Update Meters
        setTimeout(() => {
            realMeter.style.width = `${realPercentage}%`;
            fakeMeter.style.width = `${fakePercentage}%`;
        }, 100);

        // Toggle Report Button
        if (isReal) {
            reportBtn.classList.add('hidden');
        } else {
            reportBtn.classList.remove('hidden');
        }

        // Face detection warning
        let warningEl = document.getElementById('face-warning');
        if (!data.face_detected) {
            if (!warningEl) {
                warningEl = document.createElement('p');
                warningEl.id = 'face-warning';
                warningEl.style.color = '#f59e0b'; // warning color (amber)
                warningEl.style.marginTop = '1rem';
                warningEl.style.fontSize = '0.9rem';
                resultsCard.insertBefore(warningEl, resetBtn);
            }
            warningEl.textContent = '⚠️ No face detected. The prediction may be unreliable for full-body photos or backgrounds.';
            warningEl.style.display = 'block';
        } else {
            if (warningEl) warningEl.style.display = 'none';
        }
    }
});
