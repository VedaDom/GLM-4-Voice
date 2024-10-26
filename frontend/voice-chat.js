// voice-chat.js
class GLMVoiceChat {
    constructor(serverUrl) {
        this.serverUrl = serverUrl;
        this.ws = null;
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.audioInitialized = false;
        this.audioChunks = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;

        // UI Elements
        this.statusElement = document.getElementById('status');
        this.recordButton = document.getElementById('recordButton');
        this.sendButton = document.getElementById('sendButton');
        this.textInput = document.getElementById('textInput');
        this.responseArea = document.getElementById('responseArea');

        // Bind methods
        this.setupUIElements = this.setupUIElements.bind(this);
        this.handleRecordButton = this.handleRecordButton.bind(this);
        this.handleSendButton = this.handleSendButton.bind(this);
        this.updateUIState = this.updateUIState.bind(this);
    }

    updateStatus(message, isError = false) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = 'status' + (isError ? ' error' : '');
        }
        console.log(`Status: ${message}`);
    }

    async initializeAudioContext() {
        if (this.audioContext && this.audioInitialized) {
            return;
        }

        try {
            // Create AudioContext but don't resume it yet
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log('Audio context created, waiting for user interaction');
        } catch (error) {
            console.error('Failed to create audio context:', error);
            this.updateStatus('Failed to initialize audio', true);
        }
    }

    async resumeAudioContext() {
        if (!this.audioContext) {
            await this.initializeAudioContext();
        }

        try {
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            this.audioInitialized = true;
            console.log('Audio context resumed successfully');
        } catch (error) {
            console.error('Failed to resume audio context:', error);
            this.updateStatus('Failed to start audio', true);
        }
    }

    async setupWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        this.updateStatus('Connecting...');
        
        try {
            this.ws = new WebSocket(this.serverUrl);

            this.ws.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateStatus('Connected');
                this.updateUIState();
                this.startHeartbeat();
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateStatus('Disconnected', true);
                this.updateUIState();
                if (this.heartbeatInterval) {
                    clearInterval(this.heartbeatInterval);
                }
                this.reconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateStatus('Connection error', true);
            };

            this.ws.onmessage = async (event) => {
                try {
                    const response = JSON.parse(event.data);
                    await this.handleServerMessage(response);
                } catch (error) {
                    console.error('Error handling message:', error);
                }
            };
        } catch (error) {
            console.error('WebSocket setup error:', error);
            this.updateStatus('Failed to setup connection', true);
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    async reconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.updateStatus('Max reconnection attempts reached. Please refresh the page.', true);
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        this.updateStatus(`Reconnecting in ${delay/1000} seconds... (Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        await new Promise(resolve => setTimeout(resolve, delay));
        this.connect().catch(() => this.reconnect());
    }

    async handleServerMessage(response) {
        switch (response.type) {
            case 'audio_chunk':
                await this.playAudioChunk(response.data, response.sample_rate);
                break;
            
            case 'complete':
                this.updateChatHistory(response.text, response.history);
                break;
            
            case 'error':
                this.updateStatus(`Error: ${response.message}`, true);
                break;
            
            case 'pong':
                // Heartbeat response received
                break;
                
            default:
                console.warn('Unknown message type:', response.type);
        }
    }

    async playAudioChunk(audioData, sampleRate) {
        if (!this.audioInitialized) {
            await this.resumeAudioContext();
        }

        const audioBuffer = this.audioContext.createBuffer(1, audioData.length, sampleRate);
        audioBuffer.getChannelData(0).set(new Float32Array(audioData));

        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start(0);
    }

    updateChatHistory(text, history) {
        if (!this.responseArea) return;

        const historyHtml = history.map(entry => {
            if (entry.role === 'user') {
                const content = typeof entry.content === 'string' 
                    ? entry.content 
                    : '[Audio Input]';
                return `<div class="user-message message">${content}</div>`;
            } else {
                const content = typeof entry.content === 'string'
                    ? entry.content
                    : '[Audio Response]';
                return `<div class="assistant-message message">${content}</div>`;
            }
        }).join('');
        
        this.responseArea.innerHTML = historyHtml;
        this.responseArea.scrollTop = this.responseArea.scrollHeight;
    }

    async startRecording() {
        try {
            await this.resumeAudioContext();

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(this.audioChunks);
                const arrayBuffer = await audioBlob.arrayBuffer();
                
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({
                        type: 'audio',
                        params: {
                            temperature: 0.2,
                            top_p: 0.8,
                            max_new_tokens: 2000
                        }
                    }));
                    this.ws.send(arrayBuffer);
                }
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateUIState();
        } catch (error) {
            console.error('Recording error:', error);
            this.updateStatus('Failed to start recording', true);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateUIState();
        }
    }

    async sendText(text) {
        if (!text.trim()) return;
        
        try {
            await this.resumeAudioContext();
            
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'text',
                    text: text,
                    params: {
                        temperature: 0.2,
                        top_p: 0.8,
                        max_new_tokens: 2000
                    }
                }));
            } else {
                this.updateStatus('Connection lost. Please refresh the page.', true);
            }
        } catch (error) {
            console.error('Send text error:', error);
            this.updateStatus('Failed to send message', true);
        }
    }

    setupUIElements() {
        if (this.recordButton) {
            this.recordButton.addEventListener('click', async () => {
                await this.resumeAudioContext();
                this.handleRecordButton();
            });
        }
        
        if (this.sendButton && this.textInput) {
            this.sendButton.addEventListener('click', async () => {
                await this.resumeAudioContext();
                this.handleSendButton();
            });
            
            this.textInput.addEventListener('keypress', async (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    await this.resumeAudioContext();
                    this.handleSendButton();
                }
            });
        }
    }

    handleRecordButton() {
        if (!this.isRecording) {
            this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    handleSendButton() {
        const text = this.textInput.value.trim();
        if (text) {
            this.sendText(text);
            this.textInput.value = '';
        }
    }

    updateUIState() {
        if (this.recordButton) {
            this.recordButton.textContent = this.isRecording ? 'Stop Recording' : 'Start Recording';
            this.recordButton.className = `button record-button${this.isRecording ? ' recording' : ''}`;
        }

        const elements = [this.recordButton, this.sendButton, this.textInput];
        elements.forEach(element => {
            if (element) {
                element.disabled = !this.isConnected;
            }
        });
    }

    async connect() {
        try {
            await this.initializeAudioContext();
            await this.setupWebSocket();
            this.setupUIElements();
        } catch (error) {
            console.error('Connection error:', error);
            this.updateStatus('Failed to connect', true);
            throw error;
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
        if (this.mediaRecorder) {
            this.stopRecording();
        }
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioInitialized = false;
        }
        this.isConnected = false;
        this.updateUIState();
    }
}

// Initialize when the document is loaded
document.addEventListener('DOMContentLoaded', async () => {
    const voiceChat = new GLMVoiceChat(`wss://${window.location.host}/ws`);
    
    try {
        await voiceChat.connect();
    } catch (error) {
        console.error('Failed to initialize voice chat:', error);
    }

    // Handle page unload
    window.addEventListener('beforeunload', () => {
        voiceChat.disconnect();
    });
});