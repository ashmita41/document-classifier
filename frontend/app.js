// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_PREFIX = '/api/v1/documents';

// State
let currentDocumentId = null;
let currentDocumentData = null;
let processingInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initUpload();
    initResults();
});

// ============================================================================
// UPLOAD FUNCTIONALITY
// ============================================================================

function initUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');

    // Browse button click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Drop zone click
    dropZone.addEventListener('click', (e) => {
        if (e.target === dropZone || e.target.closest('.drop-zone-content')) {
            fileInput.click();
        }
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
    
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type === 'application/pdf') {
            handleFileSelect(file);
        } else {
            showToast('Please upload a PDF file', 'error');
        }
    });
}

async function handleFileSelect(file) {
    // Validate file
    if (!file.type === 'application/pdf') {
        showToast('Please upload a PDF file', 'error');
        return;
    }

    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showToast('File too large. Maximum size is 50MB', 'error');
        return;
    }
    
    // Show progress
    const progressContainer = document.getElementById('uploadProgress');
    const successContainer = document.getElementById('uploadSuccess');
    const dropZone = document.getElementById('dropZone');
    
    dropZone.style.display = 'none';
    progressContainer.style.display = 'block';
    successContainer.style.display = 'none';

    try {
        // Upload file
    const formData = new FormData();
    formData.append('file', file);
        formData.append('process_immediately', 'true');

        const response = await fetch(`${API_BASE_URL}${API_PREFIX}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        currentDocumentId = data.document_id;

        // Show success
        progressContainer.style.display = 'none';
        successContainer.style.display = 'block';

        // Start polling for status
        pollDocumentStatus(data.document_id);
        
    } catch (error) {
        console.error('Upload error:', error);
        showToast(`Upload failed: ${error.message}`, 'error');
        
        // Reset UI
        dropZone.style.display = 'block';
        progressContainer.style.display = 'none';
        successContainer.style.display = 'none';
    }
}

async function pollDocumentStatus(documentId) {
    const statusText = document.getElementById('processingStatus');
    
    processingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}${API_PREFIX}/${documentId}/status`);
            
            if (!response.ok) {
                throw new Error('Failed to get status');
            }

            const status = await response.json();
            statusText.textContent = `Status: ${status.current_stage || 'Processing'}... ${Math.round(status.progress_percentage)}%`;

            if (status.status === 'completed') {
                clearInterval(processingInterval);
                showToast('Document processed successfully!', 'success');
                
                // Wait a moment then show results
                setTimeout(() => {
                    loadDocumentResults(documentId);
                }, 1000);
            } else if (status.status === 'failed') {
                clearInterval(processingInterval);
                showToast(`Processing failed: ${status.error_message}`, 'error');
                resetUploadView();
            }
        } catch (error) {
            console.error('Status check error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

function resetUploadView() {
    document.getElementById('dropZone').style.display = 'block';
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('uploadSuccess').style.display = 'none';
}

// ============================================================================
// RESULTS VIEW
// ============================================================================

function initResults() {
    const backBtn = document.getElementById('backBtn');
    const exportBtn = document.getElementById('exportBtn');
    const copyBtn = document.getElementById('copyBtn');

    backBtn.addEventListener('click', () => {
        switchToUploadView();
    });

    exportBtn.addEventListener('click', () => {
        if (currentDocumentData) {
            downloadJSON(currentDocumentData);
        }
    });

    copyBtn.addEventListener('click', () => {
        if (currentDocumentData) {
            copyToClipboard(currentDocumentData);
        }
    });
}

async function loadDocumentResults(documentId) {
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}${API_PREFIX}/${documentId}/simplified`);
        
        if (!response.ok) {
            throw new Error('Failed to load document');
        }
        
        const data = await response.json();
        displayResults(data);
        switchToResultsView();
        
    } catch (error) {
        console.error('Error loading results:', error);
        showToast(`Failed to load results: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    // Store data for copy/download
    currentDocumentData = data;

    // Update header
    document.getElementById('docTitle').textContent = 'Document Analysis Results';
    document.getElementById('docInfo').textContent = currentDocumentId ? `Document ID: ${currentDocumentId}` : '';

    // Display raw JSON
    const contentContainer = document.getElementById('documentContent');
    contentContainer.innerHTML = '';

    // Create JSON display
    const jsonDisplay = document.createElement('pre');
    jsonDisplay.style.cssText = `
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 2rem;
        border-radius: 8px;
        overflow-x: auto;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    `;
    
    // Format JSON with syntax highlighting
    const jsonString = JSON.stringify(data, null, 2);
    jsonDisplay.textContent = jsonString;
    
    contentContainer.appendChild(jsonDisplay);
}

function createMetadataSection(metadata) {
    const section = document.createElement('div');
    section.className = 'metadata-section';

    const title = document.createElement('div');
    title.className = 'metadata-title';
    title.innerHTML = '📋 Extracted Metadata';
    section.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'metadata-grid';

    // Handle both array and object formats
    if (Array.isArray(metadata)) {
        // Array format: [{key: "", value: ""}]
        metadata.forEach(item => {
            const metadataItem = document.createElement('div');
            metadataItem.className = 'metadata-item';
            metadataItem.innerHTML = `
                <span class="metadata-key">${escapeHtml(item.key)}:</span>
                <span class="metadata-value">${escapeHtml(item.value)}</span>
            `;
            grid.appendChild(metadataItem);
        });
    } else {
        // Object format: {key: value, key2: value2}
        for (const [key, value] of Object.entries(metadata)) {
            const metadataItem = document.createElement('div');
            metadataItem.className = 'metadata-item';
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            metadataItem.innerHTML = `
                <span class="metadata-key">${escapeHtml(displayKey)}:</span>
                <span class="metadata-value">${escapeHtml(String(value))}</span>
            `;
            grid.appendChild(metadataItem);
        }
    }

    section.appendChild(grid);
    return section;
}

function createSectionElement(title, sectionData) {
    const sectionDiv = document.createElement('div');
    sectionDiv.className = 'section';

    // Header
    const header = document.createElement('div');
    header.className = 'section-header';
    header.innerHTML = `
        <h3 class="section-title">${escapeHtml(title)}</h3>
        <span class="section-badge">Section</span>
    `;
    sectionDiv.appendChild(header);

    // Content
    const content = document.createElement('div');
    content.className = 'section-content';

    // Descriptions
    if (sectionData.descriptions && sectionData.descriptions.length > 0) {
        sectionData.descriptions.forEach(desc => {
            const descDiv = document.createElement('div');
            descDiv.className = 'description';
            descDiv.textContent = desc;
            content.appendChild(descDiv);
        });
    }

    // Section-specific metadata (fields other than type, descriptions, questions)
    const sectionMetadata = {};
    for (const [key, value] of Object.entries(sectionData)) {
        if (key !== 'type' && key !== 'descriptions' && key !== 'questions') {
            sectionMetadata[key] = value;
        }
    }

    if (Object.keys(sectionMetadata).length > 0) {
        const metadataDiv = document.createElement('div');
        metadataDiv.style.marginTop = '1rem';
        
        const metadataGrid = document.createElement('div');
        metadataGrid.className = 'metadata-grid';
        
        for (const [key, value] of Object.entries(sectionMetadata)) {
            const metadataItem = document.createElement('div');
            metadataItem.className = 'metadata-item';
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            metadataItem.innerHTML = `
                <span class="metadata-key">${escapeHtml(displayKey)}:</span>
                <span class="metadata-value">${escapeHtml(String(value))}</span>
            `;
            metadataGrid.appendChild(metadataItem);
        }
        
        metadataDiv.appendChild(metadataGrid);
        content.appendChild(metadataDiv);
    }

    // Questions
    if (sectionData.questions && sectionData.questions.length > 0) {
        const qaDiv = createQASection(sectionData.questions);
        content.appendChild(qaDiv);
    }

    sectionDiv.appendChild(content);
    return sectionDiv;
}

function createQASection(questions, title = null) {
    const qaSection = document.createElement('div');
    qaSection.className = 'question-section';
    qaSection.style.marginTop = '1.5rem';

    if (title) {
        const titleEl = document.createElement('h4');
        titleEl.style.marginBottom = '1rem';
        titleEl.textContent = title;
        qaSection.appendChild(titleEl);
    }

    questions.forEach(qa => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'question';

        const questionText = document.createElement('div');
        questionText.className = 'question-text';
        questionText.innerHTML = `
            <span class="question-icon">❓</span>
            <span>${escapeHtml(qa.question)}</span>
        `;
        questionDiv.appendChild(questionText);

        if (qa.answer && qa.answer !== 'Not answered' && qa.answer !== null) {
            const answerText = document.createElement('div');
            answerText.className = 'answer-text';
            answerText.textContent = qa.answer;
            questionDiv.appendChild(answerText);
        } else {
            const answerText = document.createElement('div');
            answerText.className = 'answer-text unanswered';
            answerText.textContent = 'Not answered';
            questionDiv.appendChild(answerText);
        }

        qaSection.appendChild(questionDiv);
    });

    return qaSection;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function switchToResultsView() {
    document.getElementById('upload-view').classList.remove('active');
    document.getElementById('results-view').classList.add('active');
}

function switchToUploadView() {
    document.getElementById('results-view').classList.remove('active');
    document.getElementById('upload-view').classList.add('active');
    resetUploadView();
    currentDocumentId = null;
    currentDocumentData = null;
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s reverse';
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

function downloadJSON(data) {
    try {
        // Create downloadable JSON file
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `document_${currentDocumentId || 'analysis'}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast('Downloaded successfully!', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showToast(`Download failed: ${error.message}`, 'error');
    }
}

async function copyToClipboard(data) {
    try {
        const jsonText = JSON.stringify(data, null, 2);
        await navigator.clipboard.writeText(jsonText);
        showToast('Copied to clipboard!', 'success');
    } catch (error) {
        console.error('Copy error:', error);
        
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = JSON.stringify(data, null, 2);
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        
        try {
            document.execCommand('copy');
            showToast('Copied to clipboard!', 'success');
        } catch (err) {
            showToast('Failed to copy', 'error');
        }
        
        document.body.removeChild(textArea);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
