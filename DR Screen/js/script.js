document.addEventListener('DOMContentLoaded', () => {
    /* =======================================
       1. Navbar Scroll Effect
       ======================================= */
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.9)';
            navbar.style.boxShadow = 'none';
        }
    });

    /* =======================================
       2. Image Upload Demo Logic
       ======================================= */
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    
    const previewArea = document.getElementById('preview-area');
    const previewImg = document.getElementById('preview-img');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resetBtn = document.getElementById('reset-btn');
    const scanLine = document.getElementById('scan-line');
    
    const resultArea = document.getElementById('result-area');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultContent = document.getElementById('result-content');

    // Trigger file input dialog when browse button is clicked
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection from input
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    // Handle Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files[0]) {
            handleFile(files[0]);
        }
    });

    // Process the selected/dropped file
    function handleFile(file) {
        // Validate it's an image
        if (!file.type.match('image.*')) {
            alert('Please upload an image file (JPEG, PNG).');
            return;
        }

        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Set image source and show preview area
            previewImg.src = e.target.result;
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
            resultArea.classList.add('hidden');
            resultContent.classList.add('hidden');
            scanLine.classList.add('hidden');
        }
        
        reader.readAsDataURL(file);
    }

    // Reset workflow
    resetBtn.addEventListener('click', () => {
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resultArea.classList.add('hidden');
        fileInput.value = ''; // Reset file input
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Analyze Image';
    });

    // Simulate AI Analysis
    analyzeBtn.addEventListener('click', () => {
        // UI State: Disable button, show scanning animation
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = 'Analyzing...';
        scanLine.classList.remove('hidden');
        
        resultArea.classList.remove('hidden');
        loadingSpinner.classList.remove('hidden');
        resultContent.classList.add('hidden');

        // Simulate network/processing delay (3 seconds)
        setTimeout(() => {
            scanLine.classList.add('hidden');
            loadingSpinner.classList.add('hidden');
            resultContent.classList.remove('hidden');
            analyzeBtn.innerHTML = 'Analysis Complete';
            
            // Scroll to results smoothly
            resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 3000);
    });
});
