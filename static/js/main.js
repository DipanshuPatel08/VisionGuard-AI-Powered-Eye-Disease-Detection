document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (!uploadForm) return;

    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const selectImageBtn = document.getElementById('select-image-btn');
    const processImageBtn = document.getElementById('process-image-btn');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');
    const filenameDisplay = document.getElementById('filename-display');

    // Trigger file input when the "Select Image" button is clicked
    selectImageBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // Drag and Drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files; // Important for form submission
            handleFile(file);
        }
    });
    
    // Also allow clicking the whole area to select a file
    uploadArea.addEventListener('click', (e) => {
        // Prevent triggering if the button was clicked
        if (e.target !== selectImageBtn) {
            fileInput.click();
        }
    });

    function handleFile(file) {
        // Check if the file is an image
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (jpg, jpeg, png).');
            return;
        }

        // Show image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            imagePreviewContainer.style.display = 'block';
            filenameDisplay.textContent = file.name;
            uploadArea.style.display = 'none'; // Hide the upload box
        }
        reader.readAsDataURL(file);

        // Enable the process button
        processImageBtn.disabled = false;
    }
});
