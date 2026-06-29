/**
 * Client Configurations for the API endpoints using relative routing.
 */
const HEALTH_URL = '/api/v1/health';
const UPLOAD_URL = '/api/v1/documents/upload';
const CHAT_URL = '/api/v1/chat';

/**
 * Cache list of indexed documents.
 * @type {Array<{name: string, chunkCount: number}>}
 */
let indexedDocs = [];

// DOM Elements
const apiStatusBadge = document.getElementById('api-status');
const dbStatusBadge = document.getElementById('db-status');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const uploadPercent = document.getElementById('upload-percent');
const filenameDisplay = document.getElementById('filename-display');
const progressStatusText = document.getElementById('status-text');
const documentsList = document.getElementById('documents-list');
const terminalLog = document.getElementById('terminal-log');
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-button');
const typingIndicator = document.getElementById('typing-indicator');

// Event listener for DOM Content Loaded
window.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    setupUploadHandlers();
    setupChatHandler();
    
    // Periodically query connection status
    setInterval(checkBackendHealth, 15000);
});

/**
 * Appends a log line entry to the Agent Log Trace Terminal.
 * @param {string} message - The log message to display.
 * @param {string} [type='system'] - Log type formatting class.
 */
function logToTerminal(message, type = 'system') {
    const timestamp = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `terminal-line ${type}-line`;
    line.innerHTML = `<span style="color: #64748b;">[${timestamp}]</span> ${message}`;
    
    if (terminalLog) {
        terminalLog.appendChild(line);
        terminalLog.scrollTop = terminalLog.scrollHeight;
    }
}

/**
 * Validates backend server and database availability asynchronously.
 * Updates system status indicators.
 * @returns {Promise<void>}
 */
async function checkBackendHealth() {
    try {
        const response = await fetch(HEALTH_URL);
        if (response.ok) {
            const healthData = await response.json();
            
            // FastAPI connection active
            updateStatusBadge(apiStatusBadge, 'online', 'FastAPI API: Online');
            
            // Validate MongoDB connectivity state
            const dbStatus = healthData.database;
            if (dbStatus === 'healthy' || dbStatus === 'connected') {
                updateStatusBadge(dbStatusBadge, 'online', 'MongoDB Atlas: Connected');
            } else if (dbStatus === 'unhealthy' || dbStatus === 'disconnected') {
                updateStatusBadge(dbStatusBadge, 'degraded', 'MongoDB Atlas: Degraded');
                logToTerminal(`[Database] MongoDB connection health check reports degraded status.`, 'warning');
            } else {
                updateStatusBadge(dbStatusBadge, 'offline', 'MongoDB Atlas: Offline');
            }
        } else {
            console.error('Health check endpoint returned non-OK status:', response.status);
            setOfflineStatus();
        }
    } catch (error) {
        console.error('Health check fetch connection failed:', error);
        setOfflineStatus();
    }
}

/**
 * Updates status indicator badge classes and text.
 * @param {HTMLElement} badge - The badge wrapper element.
 * @param {'online'|'offline'|'degraded'} state - Connection state.
 * @param {string} labelText - Text display status string.
 */
function updateStatusBadge(badge, state, labelText) {
    if (!badge) return;
    const dot = badge.querySelector('.status-dot');
    const label = badge.querySelector('.status-label');
    if (dot && label) {
        dot.className = `status-dot ${state}`;
        label.textContent = labelText;
    }
}

/**
 * Updates status panels to Offline.
 */
function setOfflineStatus() {
    updateStatusBadge(apiStatusBadge, 'offline', 'FastAPI API: Offline');
    updateStatusBadge(dbStatusBadge, 'offline', 'MongoDB Atlas: Offline');
}

/**
 * Attaches event listeners to document dropzone and file selection inputs.
 */
function setupUploadHandlers() {
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        ['dragleave', 'dragend'].forEach(event => {
            dropZone.addEventListener(event, () => {
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }
}

/**
 * Sends a PDF file stream using XMLHttpRequests to monitor upload progress.
 * @param {File} file - PDF document file.
 */
function handleFileUpload(file) {
    if (!file.name.endsWith('.pdf')) {
        logToTerminal('[Upload] Rejection: Only PDF files can be parsed.', 'warning');
        alert('Rejection: Only PDF documents are supported.');
        return;
    }

    if (dropZone) dropZone.style.display = 'none';
    if (progressContainer) progressContainer.style.display = 'block';
    if (filenameDisplay) filenameDisplay.innerHTML = `<i class="fa-solid fa-file-pdf"></i> ${file.name}`;
    if (progressFill) progressFill.style.width = '0%';
    if (uploadPercent) uploadPercent.textContent = '0%';
    if (progressStatusText) progressStatusText.textContent = 'Uploading PDF stream...';
    
    logToTerminal(`[Upload] Starting upload pipeline for: ${file.name}`, 'system');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', UPLOAD_URL, true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percentComplete = Math.round((e.loaded / e.total) * 100);
            if (progressFill) progressFill.style.width = `${percentComplete}%`;
            if (uploadPercent) uploadPercent.textContent = `${percentComplete}%`;
            if (percentComplete === 100) {
                if (progressStatusText) progressStatusText.textContent = 'FastAPI server extracting text & initializing embedding vectors...';
                logToTerminal('[Upload] File upload complete. Processing server-side text-splitting...', 'system');
            }
        }
    };

    xhr.onload = () => {
        if (xhr.status === 200) {
            const response = JSON.parse(xhr.responseText);
            logToTerminal(`[Upload] SUCCESS: Chunks created & vectorized. Chunks = ${response.chunks_vectorized}`, 'success');
            addDocumentToCorpus(response.filename, response.chunks_vectorized);
            if (progressStatusText) progressStatusText.textContent = 'Ingestion complete!';
            setTimeout(resetUploadUI, 2000);
        } else {
            let errorMsg = 'Failed to extract text or insert vectors.';
            try {
                const err = JSON.parse(xhr.responseText);
                errorMsg = err.detail || errorMsg;
            } catch (e) {}
            
            console.error('File upload failed with status:', xhr.status, xhr.responseText);
            logToTerminal(`[Upload] Pipeline error: ${errorMsg}`, 'warning');
            if (progressStatusText) progressStatusText.textContent = 'Ingestion failed!';
            setTimeout(resetUploadUI, 3000);
        }
    };

    xhr.onerror = (err) => {
        console.error('File upload network or CORS connection failure:', err);
        logToTerminal('[Upload] Network connection failed during stream upload.', 'warning');
        if (progressStatusText) progressStatusText.textContent = 'Network upload failure.';
        setTimeout(resetUploadUI, 3000);
    };

    const formData = new FormData();
    formData.append('file', file);
    xhr.send(formData);
}

/**
 * Resets file upload panels back to default display states.
 */
function resetUploadUI() {
    if (progressContainer) progressContainer.style.display = 'none';
    if (dropZone) dropZone.style.display = 'flex';
    if (fileInput) fileInput.value = '';
}

/**
 * Renders the uploaded document meta inside the local index catalog panel.
 * @param {string} name - The uploaded file name.
 * @param {number} chunkCount - Partition chunk count.
 */
function addDocumentToCorpus(name, chunkCount) {
    indexedDocs.push({ name, chunkCount });
    
    if (documentsList) {
        const emptyMsg = documentsList.querySelector('.empty-docs-message');
        if (emptyMsg) emptyMsg.remove();
        
        const item = document.createElement('div');
        item.className = 'doc-item';
        item.innerHTML = `
            <div class="doc-info">
                <i class="fa-solid fa-file-pdf"></i>
                <div>
                    <div class="doc-name" title="${name}">${name}</div>
                    <div class="doc-meta">Status: Active</div>
                </div>
            </div>
            <div class="doc-chunks">${chunkCount} chunks</div>
        `;
        documentsList.appendChild(item);
    }
}

/**
 * Initializes form submit handlers linking user queries to RAG executions.
 */
function setupChatHandler() {
    if (chatForm) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = queryInput.value.trim();
            if (!query) return;

            queryInput.value = '';
            queryInput.disabled = true;
            sendButton.disabled = true;

            appendMessage(query, 'user');
            
            if (typingIndicator) typingIndicator.style.display = 'flex';
            if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;

            const welcomePanel = chatHistory.querySelector('.welcome-chat-message');
            if (welcomePanel) welcomePanel.remove();

            simulateAgentTraceLogs(query);

            try {
                const response = await fetch(CHAT_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

                if (response.ok) {
                    const data = await response.json();
                    if (typingIndicator) typingIndicator.style.display = 'none';
                    logToTerminal('[System] RAG response generated. Appending citations.', 'success');
                    appendMessage(data.answer, 'bot', data.citations, data.web_fallback_triggered);
                } else {
                    let errorMsg = 'Failed to execute agent workflow.';
                    try {
                        const err = await response.json();
                        errorMsg = err.detail || errorMsg;
                    } catch (e) {}
                    
                    console.error('Chat query post failed with status:', response.status, errorMsg);
                    if (typingIndicator) typingIndicator.style.display = 'none';
                    logToTerminal(`[System] RAG node crash: ${errorMsg}`, 'warning');
                    appendMessage(`Error: ${errorMsg}`, 'bot');
                }
            } catch (error) {
                console.error('Chat query connection failed:', error);
                if (typingIndicator) typingIndicator.style.display = 'none';
                logToTerminal(`[System] Network communication failure: ${error.message}`, 'warning');
                appendMessage('Network error communicating with the self-correcting RAG server. Make sure FastAPI is running on port 8000.', 'bot');
            } finally {
                queryInput.disabled = false;
                sendButton.disabled = false;
                queryInput.focus();
                if (chatHistory) chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        });
    }
}

/**
 * Animates terminal execution trace steps for agent search operations.
 * @param {string} query - The search query string.
 */
function simulateAgentTraceLogs(query) {
    logToTerminal(`[User Query] Initializing search state: "${query}"`, 'system');
    
    setTimeout(() => {
        logToTerminal('[Retrieve Node] Vectorizing query using local BAAI/bge-small-en-v1.5 model...', 'node');
    }, 300);
    
    setTimeout(() => {
        logToTerminal('[Retrieve Node] Querying MongoDB Atlas document chunks vector index...', 'node');
    }, 800);
    
    setTimeout(() => {
        logToTerminal('[Grade Node] Reranking retrieved chunks using BAAI/bge-reranker-base cross-encoder...', 'node');
    }, 1500);

    setTimeout(() => {
        logToTerminal('[Grade Node] Calculating chunk relevance score thresholds...', 'node');
    }, 2000);
}

/**
 * Creates and appends chat message bubbles (User or Bot) to the conversation history.
 * @param {string} text - Message context text.
 * @param {'user'|'bot'} sender - Message author identifier.
 * @param {Array<Object>} [citations=[]] - Optional references payload.
 * @param {boolean} [webFallbackTriggered=false] - Web fallback query router indicator status.
 */
function appendMessage(text, sender, citations = [], webFallbackTriggered = false) {
    if (!chatHistory) return;
    
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${sender}-msg`;
    
    const avatar = document.createElement('div');
    avatar.className = `${sender}-avatar`;
    avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    
    let formattedText = text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
        
    bubble.innerHTML = `<p>${formattedText}</p>`;
    
    if (sender === 'bot' && citations && citations.length > 0) {
        if (webFallbackTriggered) {
            logToTerminal('[Router] WARNING: Chunks failed grading. Web fallback results active.', 'warning');
        } else {
            logToTerminal('[Router] Document context verified. Direct generation active.', 'success');
        }

        const citationBox = document.createElement('div');
        citationBox.className = 'citations-container';
        citationBox.innerHTML = `<span class="citations-label"><i class="fa-solid fa-circle-info"></i> Sources & Citations Verified (${citations.length})</span>`;
        
        citations.forEach(cite => {
            const isWeb = cite.source.startsWith('http') || cite.source.includes('Web Search');
            const card = document.createElement('div');
            card.className = `citation-card ${isWeb ? 'web-source' : ''}`;
            
            const scorePercent = (cite.relevance_score * 100).toFixed(1);
            
            card.innerHTML = `
                <div class="citation-meta">
                    <span class="citation-title" title="${cite.source}">
                        <i class="${isWeb ? 'fa-solid fa-globe' : 'fa-solid fa-file-pdf'}"></i> ${cite.source}
                    </span>
                    <span class="citation-score">
                        ${isWeb ? 'Web Fallback' : `Relevance: ${scorePercent}%`}
                    </span>
                </div>
                <div class="citation-text">
                    "${cite.content}"
                </div>
            `;
            citationBox.appendChild(card);
        });
        bubble.appendChild(citationBox);
    }
    
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}
