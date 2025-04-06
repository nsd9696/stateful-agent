// Generate a unique session ID
const sessionId = Math.random().toString(36).substring(2, 15);

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const filesList = document.getElementById('files-list');
const loadingIndicator = document.getElementById('loading-indicator');
const sessionIdElement = document.getElementById('session-id');

// Display session ID
sessionIdElement.textContent = sessionId;

// Add message to chat
function addMessage(message, isUser = false) {
    // Create message container
    const messageContainer = document.createElement('div');
    messageContainer.className = `flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`;
    
    // Create message bubble
    const messageDiv = document.createElement('div');
    messageDiv.className = `p-4 rounded-lg max-w-[80%] ${
        isUser 
            ? 'bg-blue-500 text-white rounded-br-none' 
            : 'bg-gray-100 text-gray-800 rounded-bl-none'
    }`;
    
    // Add timestamp
    const timestamp = document.createElement('div');
    timestamp.className = `text-xs mb-1 ${isUser ? 'text-blue-100' : 'text-gray-500'}`;
    timestamp.textContent = new Date().toLocaleTimeString();
    
    // Add message content
    const content = document.createElement('div');
    content.className = 'whitespace-pre-wrap';
    content.textContent = message;
    
    messageDiv.appendChild(timestamp);
    messageDiv.appendChild(content);
    messageContainer.appendChild(messageDiv);
    chatMessages.appendChild(messageContainer);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Clear chat history
function clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        chatMessages.innerHTML = `
            <div class="flex justify-start mb-4">
                <div class="p-4 rounded-lg bg-gray-100 text-gray-800 rounded-bl-none max-w-[80%]">
                    <div class="text-xs text-gray-500 mb-1">${new Date().toLocaleTimeString()}</div>
                    <div class="whitespace-pre-wrap">👋 Welcome! I'm your Stateful Agent. How can I help you today?</div>
                </div>
            </div>
        `;
    }
}

// Show loading indicator
function showLoading() {
    loadingIndicator.classList.remove('hidden');
}

// Hide loading indicator
function hideLoading() {
    loadingIndicator.classList.add('hidden');
}

// Send message to backend
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessage(message, true);
    userInput.value = '';
    showLoading();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        const data = await response.json();
        
        if (response.ok) {
            addMessage(data.response);
            // Refresh files list after successful message
            loadFiles();
        } else {
            addMessage('Error: ' + data.error);
        }
    } catch (error) {
        addMessage('Error: Failed to send message');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

// Load files list
async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        const data = await response.json();
        
        filesList.innerHTML = '';
        if (data.files.length === 0) {
            filesList.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="fas fa-folder-open text-4xl mb-2"></i>
                    <p>No files generated yet</p>
                </div>
            `;
            return;
        }

        data.files.forEach(file => {
            const fileDiv = document.createElement('div');
            fileDiv.className = 'p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer';
            
            const fileName = document.createElement('div');
            fileName.className = 'font-medium text-gray-800';
            fileName.textContent = file.split('/').pop();
            
            const filePath = document.createElement('div');
            filePath.className = 'text-sm text-gray-500';
            filePath.textContent = file;
            
            fileDiv.appendChild(fileName);
            fileDiv.appendChild(filePath);
            filesList.appendChild(fileDiv);
        });
    } catch (error) {
        console.error('Error loading files:', error);
        filesList.innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fas fa-exclamation-circle text-4xl mb-2"></i>
                <p>Failed to load files</p>
            </div>
        `;
    }
}

// Event Listeners
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Initial load
loadFiles(); 