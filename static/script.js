// Set API URL (change this to your actual backend URL when deployed)
const API_URL = 'http://localhost:5000/api';

// State
let currentChatId = null;

// DOM Elements
const newChatBtn = document.getElementById('new-chat-btn');
const chatList = document.getElementById('chat-list');
const messagesContainer = document.getElementById('messages');
const welcomeMessage = document.getElementById('welcome-message');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Event Listeners
document.addEventListener('DOMContentLoaded', loadChats);
newChatBtn.addEventListener('click', createNewChat);
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Functions
async function loadChats() {
    try {
        const response = await fetch(`${API_URL}/chats`);
        const chats = await response.json();
        
        chatList.innerHTML = '';
        
        chats.forEach(chat => {
            const chatItem = document.createElement('div');
            chatItem.className = 'chat-item';
            chatItem.dataset.id = chat.id;
            chatItem.textContent = chat.preview;
            chatItem.addEventListener('click', () => loadChat(chat.id));
            chatList.appendChild(chatItem);
        });
    } catch (error) {
        console.error('Error loading chats:', error);
    }
}

async function createNewChat() {
    try {
        const response = await fetch(`${API_URL}/chats`, {
            method: 'POST'
        });
        const data = await response.json();
        currentChatId = data.chat_id;
        
        // Clear messages
        messagesContainer.innerHTML = '';
        messagesContainer.classList.remove('d-none');
        welcomeMessage.classList.add('d-none');
        
        // Update chat list
        loadChats();
        
        // Set active chat
        setActiveChat(currentChatId);
        
        // Focus on input
        userInput.focus();
    } catch (error) {
        console.error('Error creating new chat:', error);
    }
}

async function loadChat(chatId) {
    try {
        const response = await fetch(`${API_URL}/chats/${chatId}`);
        const messages = await response.json();
        
        currentChatId = chatId;
        
        // Clear messages
        messagesContainer.innerHTML = '';
        messagesContainer.classList.remove('d-none');
        welcomeMessage.classList.add('d-none');
        
        // Render messages
        messages.forEach(message => {
            renderMessage(message);
        });
        
        // Set active chat
        setActiveChat(chatId);
        
        // Focus on input
        userInput.focus();
    } catch (error) {
        console.error('Error loading chat:', error);
    }
}

function setActiveChat(chatId) {
    // Remove active class from all chat items
    document.querySelectorAll('.chat-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add active class to current chat
    const chatItem = document.querySelector(`.chat-item[data-id="${chatId}"]`);
    if (chatItem) {
        chatItem.classList.add('active');
    }
}

async function sendMessage() {
    const message = userInput.value.trim();
    
    if (!message) return;
    
    // If no chat is active, create a new one
    if (!currentChatId) {
        await createNewChat();
    }
    
    // Clear input
    userInput.value = '';
    
    // Add user message to UI
    const userMessage = {
        role: 'user',
        content: message
    };
    renderMessage(userMessage);
    
    // Add loading message
    const loadingId = 'loading-' + Date.now();
    const loadingHTML = `
        <div class="message assistant-message" id="${loadingId}">
            <div class="message-content">
                <p><i class="bi bi-three-dots"></i> Thinking...</p>
            </div>
        </div>
    `;
    messagesContainer.insertAdjacentHTML('beforeend', loadingHTML);
    
    try {
        // Send message to API
        const response = await fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                chat_id: currentChatId,
                query: message
            })
        });
        
        // Remove loading message
        document.getElementById(loadingId).remove();
        
        const data = await response.json();
        
        // Add assistant message to UI
        const assistantMessage = {
            role: 'assistant',
            content: data.answer,
            sources: data.sources
        };
        renderMessage(assistantMessage);
        
        // Update chat list
        loadChats();
    } catch (error) {
        console.error('Error sending message:', error);
        // Remove loading message
        document.getElementById(loadingId).remove();
        
        // Add error message
        const errorHTML = `
            <div class="message assistant-message">
                <div class="message-content">
                    <p class="text-danger">Error: Could not get a response. Please try again.</p>
                </div>
            </div>
        `;
        messagesContainer.insertAdjacentHTML('beforeend', errorHTML);
    }
}

function renderMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const paragraph = document.createElement('p');
    paragraph.textContent = message.content;
    contentDiv.appendChild(paragraph);
    
    // Add sources if they exist
    if (message.sources && message.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const sourcesText = document.createElement('p');
        sourcesText.textContent = 'Sources:';
        sourcesDiv.appendChild(sourcesText);
        
        const sourcesList = document.createElement('ul');
        message.sources.forEach(source => {
            const sourceItem = document.createElement('li');
            const sourceLink = document.createElement('a');
            sourceLink.href = source;
            sourceLink.target = '_blank';
            sourceLink.className = 'source-link';
            sourceLink.textContent = source;
            sourceItem.appendChild(sourceLink);
            sourcesList.appendChild(sourceItem);
        });
        
        sourcesDiv.appendChild(sourcesList);
        contentDiv.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}