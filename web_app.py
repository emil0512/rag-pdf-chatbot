import os
import shutil
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import uuid

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store sessions
sessions = {}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

class ExportRequest(BaseModel):
    session_id: str

def generate_suggested_questions(context, num_questions=5):
    """Generate suggested questions based on document context"""
    prompt = f"""Based on the following document context, generate {num_questions} questions that a user might want to ask.

Context preview:
{context[:2000]}

Generate {num_questions} questions. Return only the questions, one per line, numbered 1-{num_questions}.
Make sure the questions are specific to the document content.
"""
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates questions about documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        questions_text = completion.choices[0].message.content
        # Parse questions
        questions = []
        for line in questions_text.strip().split('\n'):
            # Remove numbering like "1. " or "1) "
            line = line.strip()
            if line and any(c.isdigit() for c in line[:3]):
                # Extract the question part after the number
                parts = line.split('.', 1)
                if len(parts) > 1:
                    questions.append(parts[1].strip())
                else:
                    parts = line.split(')', 1)
                    if len(parts) > 1:
                        questions.append(parts[1].strip())
                    else:
                        questions.append(line)
            elif line:
                questions.append(line)
        
        # If we got fewer questions than requested, pad with generic ones
        while len(questions) < num_questions:
            questions.append(f"Can you provide more details about the document?")
        
        return questions[:num_questions]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return [
            "What is the main topic of this document?",
            "Summarize the key points of this document.",
            "What are the most important facts in this document?",
            "Who is the target audience for this document?",
            "What conclusions are presented in this document?"
        ]

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Chatbot - RAG AI</title>
        <style>
            /* ===== CSS VARIABLES ===== */
            :root {
                --bg-primary: #f0f2f5;
                --bg-secondary: #ffffff;
                --bg-chat: #f7fafc;
                --bg-header: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --text-primary: #2d3748;
                --text-secondary: #4a5568;
                --text-muted: #718096;
                --border-color: #e2e8f0;
                --shadow: 0 20px 60px rgba(0,0,0,0.15);
                --user-msg: #667eea;
                --assistant-msg: #ffffff;
                --input-bg: #ffffff;
                --card-bg: #ffffff;
            }
            
            [data-theme="dark"] {
                --bg-primary: #1a202c;
                --bg-secondary: #2d3748;
                --bg-chat: #1a202c;
                --bg-header: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                --text-primary: #e2e8f0;
                --text-secondary: #a0aec0;
                --text-muted: #718096;
                --border-color: #4a5568;
                --shadow: 0 20px 60px rgba(0,0,0,0.5);
                --user-msg: #667eea;
                --assistant-msg: #2d3748;
                --input-bg: #2d3748;
                --card-bg: #2d3748;
            }
            
            /* ===== RESET ===== */
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: var(--bg-primary);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                transition: background 0.3s;
            }
            
            /* ===== CONTAINER ===== */
            .container {
                background: var(--bg-secondary);
                border-radius: 20px;
                box-shadow: var(--shadow);
                width: 100%;
                max-width: 1100px;
                height: 95vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                transition: background 0.3s;
                position: relative;
            }
            
            /* ===== HEADER ===== */
            .header {
                background: var(--bg-header);
                color: white;
                padding: 15px 25px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-shrink: 0;
            }
            .header-left { display: flex; align-items: center; gap: 15px; }
            .header h1 { font-size: 20px; font-weight: 600; }
            .header .badge {
                background: rgba(255,255,255,0.2);
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 500;
            }
            .header-actions { display: flex; gap: 10px; align-items: center; }
            
            /* ===== BUTTONS ===== */
            .btn {
                padding: 8px 18px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
                font-size: 13px;
                display: inline-flex;
                align-items: center;
                gap: 6px;
            }
            .btn-primary { background: #667eea; color: white; }
            .btn-primary:hover { background: #5a67d8; transform: translateY(-2px); box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
            .btn-success { background: #48bb78; color: white; }
            .btn-success:hover { background: #38a169; transform: translateY(-2px); }
            .btn-danger { background: #fc8181; color: white; }
            .btn-danger:hover { background: #f56565; transform: translateY(-2px); }
            .btn-outline { background: transparent; border: 2px solid rgba(255,255,255,0.5); color: white; }
            .btn-outline:hover { background: rgba(255,255,255,0.15); }
            .btn-sm { padding: 5px 12px; font-size: 12px; }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
            
            /* ===== THEME TOGGLE ===== */
            .theme-toggle {
                background: rgba(255,255,255,0.15);
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                cursor: pointer;
                color: white;
                font-size: 18px;
                transition: background 0.3s;
            }
            .theme-toggle:hover { background: rgba(255,255,255,0.25); }
            
            /* ===== UPLOAD SECTION ===== */
            .upload-section {
                padding: 15px 25px;
                border-bottom: 1px solid var(--border-color);
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
                flex-shrink: 0;
            }
            .upload-btn-wrapper {
                position: relative;
                overflow: hidden;
                display: inline-block;
            }
            .upload-btn-wrapper input[type=file] {
                position: absolute;
                left: 0;
                top: 0;
                opacity: 0;
                width: 100%;
                height: 100%;
                cursor: pointer;
            }
            .file-info {
                font-size: 13px;
                color: var(--text-secondary);
                background: var(--bg-primary);
                padding: 5px 14px;
                border-radius: 6px;
                transition: background 0.3s;
            }
            .status {
                font-size: 13px;
                color: var(--text-muted);
                transition: color 0.3s;
            }
            .status.success { color: #48bb78; }
            .status.error { color: #fc8181; }
            
            /* ===== DASHBOARD ===== */
            .dashboard {
                background: var(--bg-primary);
                padding: 12px 25px;
                border-bottom: 1px solid var(--border-color);
                display: none;
                flex-shrink: 0;
                transition: background 0.3s;
            }
            .dashboard.active { display: flex; }
            .dashboard-stats {
                display: flex;
                gap: 25px;
                flex-wrap: wrap;
                width: 100%;
            }
            .stat-item {
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--text-secondary);
                font-size: 13px;
            }
            .stat-item .icon { font-size: 18px; }
            .stat-item .value {
                font-weight: 700;
                color: var(--text-primary);
                font-size: 16px;
            }
            .stat-item .label { color: var(--text-muted); font-size: 12px; }
            
            /* ===== SUGGESTED QUESTIONS ===== */
            .suggested-container {
                padding: 10px 25px;
                border-bottom: 1px solid var(--border-color);
                display: none;
                flex-shrink: 0;
                gap: 8px;
                flex-wrap: wrap;
            }
            .suggested-container.active { display: flex; }
            .suggested-label {
                font-size: 12px;
                color: var(--text-muted);
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 6px;
                margin-right: 8px;
            }
            .suggested-btn {
                padding: 4px 14px;
                border: 1px solid var(--border-color);
                border-radius: 20px;
                background: var(--bg-secondary);
                color: var(--text-secondary);
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .suggested-btn:hover {
                background: #667eea;
                color: white;
                border-color: #667eea;
                transform: translateY(-1px);
            }
            
            /* ===== CHAT AREA ===== */
            .chat-area {
                flex: 1;
                overflow-y: auto;
                padding: 20px 25px;
                background: var(--bg-chat);
                transition: background 0.3s;
            }
            .chat-area::-webkit-scrollbar { width: 6px; }
            .chat-area::-webkit-scrollbar-track { background: var(--bg-chat); }
            .chat-area::-webkit-scrollbar-thumb { background: #cbd5e0; border-radius: 3px; }
            
            .message {
                margin-bottom: 12px;
                max-width: 80%;
                padding: 12px 18px;
                border-radius: 12px;
                line-height: 1.6;
                word-wrap: break-word;
                animation: fadeIn 0.3s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user {
                background: var(--user-msg);
                color: white;
                margin-left: auto;
                border-bottom-right-radius: 4px;
            }
            .message.assistant {
                background: var(--assistant-msg);
                color: var(--text-primary);
                border: 1px solid var(--border-color);
                border-bottom-left-radius: 4px;
            }
            .message .timestamp {
                font-size: 10px;
                color: var(--text-muted);
                margin-top: 6px;
                display: block;
            }
            .message.user .timestamp { color: rgba(255,255,255,0.7); }
            .message .sources {
                margin-top: 8px;
                font-size: 12px;
                color: var(--text-muted);
                border-top: 1px solid var(--border-color);
                padding-top: 8px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 15px;
                color: var(--text-muted);
            }
            .loading.active { display: block; }
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid var(--border-color);
                border-radius: 50%;
                border-top-color: #667eea;
                animation: spin 0.8s linear infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            
            /* ===== INPUT AREA ===== */
            .input-area {
                padding: 15px 25px;
                border-top: 1px solid var(--border-color);
                display: flex;
                gap: 10px;
                background: var(--bg-secondary);
                flex-shrink: 0;
                transition: background 0.3s;
            }
            .input-area input {
                flex: 1;
                padding: 12px 18px;
                border: 2px solid var(--border-color);
                border-radius: 10px;
                font-size: 14px;
                outline: none;
                transition: border 0.3s;
                background: var(--input-bg);
                color: var(--text-primary);
            }
            .input-area input:focus { border-color: #667eea; }
            .input-area input:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            /* ===== RESPONSIVE ===== */
            @media (max-width: 768px) {
                body { padding: 10px; }
                .container { height: 98vh; border-radius: 12px; }
                .header { padding: 12px 15px; flex-wrap: wrap; gap: 8px; }
                .header h1 { font-size: 16px; }
                .upload-section { padding: 10px 15px; flex-direction: column; align-items: stretch; }
                .upload-btn-wrapper { width: 100%; }
                .upload-btn-wrapper .btn { width: 100%; justify-content: center; }
                .file-info { text-align: center; }
                .dashboard-stats { gap: 12px; }
                .stat-item { font-size: 12px; }
                .chat-area { padding: 12px 15px; }
                .message { max-width: 90%; }
                .input-area { padding: 10px 15px; flex-wrap: wrap; }
                .input-area input { min-width: 0; }
                .suggested-container { padding: 8px 15px; }
                .header-actions { gap: 6px; }
            }
        </style>
    </head>
    <body>
        <div class="container" id="app">
            <!-- HEADER -->
            <div class="header">
                <div class="header-left">
                    <h1>📄 PDF Chatbot</h1>
                    <span class="badge">RAG + Groq AI</span>
                </div>
                <div class="header-actions">
                    <button class="btn btn-outline btn-sm" id="exportBtn" disabled>📥 Export Chat</button>
                    <button class="theme-toggle" id="themeToggle" title="Toggle Dark Mode">🌙</button>
                </div>
            </div>
            
            <!-- UPLOAD -->
            <div class="upload-section">
                <div class="upload-btn-wrapper">
                    <button class="btn btn-primary">📤 Upload PDF</button>
                    <input type="file" id="fileInput" accept=".pdf">
                </div>
                <span class="file-info" id="fileInfo">No file uploaded</span>
                <span class="status" id="status"></span>
                <button class="btn btn-danger btn-sm" id="clearBtn" style="display:none;">🗑️ Clear</button>
            </div>
            
            <!-- DASHBOARD -->
            <div class="dashboard" id="dashboard">
                <div class="dashboard-stats" id="stats">
                    <div class="stat-item">
                        <span class="icon">📄</span>
                        <span class="value" id="statPages">0</span>
                        <span class="label">pages</span>
                    </div>
                    <div class="stat-item">
                        <span class="icon">📊</span>
                        <span class="value" id="statChunks">0</span>
                        <span class="label">chunks</span>
                    </div>
                    <div class="stat-item">
                        <span class="icon">📏</span>
                        <span class="value" id="statSize">0 KB</span>
                        <span class="label">size</span>
                    </div>
                    <div class="stat-item">
                        <span class="icon">💬</span>
                        <span class="value" id="statMessages">0</span>
                        <span class="label">messages</span>
                    </div>
                </div>
            </div>
            
            <!-- SUGGESTED QUESTIONS -->
            <div class="suggested-container" id="suggestedContainer">
                <span class="suggested-label">💡 Try asking:</span>
                <div id="suggestedQuestions"></div>
            </div>
            
            <!-- CHAT -->
            <div class="chat-area" id="chatArea">
                <div class="message assistant">
                    👋 Hello! Upload a PDF and I'll answer questions about it.
                </div>
                <div class="loading" id="loading">
                    <div class="spinner"></div> Generating answer...
                </div>
            </div>
            
            <!-- INPUT -->
            <div class="input-area">
                <input type="text" id="questionInput" placeholder="Ask a question about your PDF..." disabled>
                <button class="btn btn-success" id="sendBtn" disabled>Send</button>
            </div>
        </div>

        <script>
            // ===== STATE =====
            let sessionId = null;
            let isProcessing = false;
            let messageCount = 0;
            let chatHistory = [];
            let documentStats = { pages: 0, chunks: 0, size: 0 };
            let isDark = false;
            
            // ===== DOM REFS =====
            const chatArea = document.getElementById('chatArea');
            const questionInput = document.getElementById('questionInput');
            const sendBtn = document.getElementById('sendBtn');
            const fileInput = document.getElementById('fileInput');
            const fileInfo = document.getElementById('fileInfo');
            const status = document.getElementById('status');
            const clearBtn = document.getElementById('clearBtn');
            const loading = document.getElementById('loading');
            const dashboard = document.getElementById('dashboard');
            const exportBtn = document.getElementById('exportBtn');
            const themeToggle = document.getElementById('themeToggle');
            const suggestedContainer = document.getElementById('suggestedContainer');
            const suggestedQuestions = document.getElementById('suggestedQuestions');
            
            // Stats elements
            const statPages = document.getElementById('statPages');
            const statChunks = document.getElementById('statChunks');
            const statSize = document.getElementById('statSize');
            const statMessages = document.getElementById('statMessages');

            // ===== THEME =====
            themeToggle.addEventListener('click', () => {
                isDark = !isDark;
                document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
                themeToggle.textContent = isDark ? '☀️' : '🌙';
            });

            // ===== ADD MESSAGE =====
            function addMessage(role, content, timestamp = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + role;
                
                const time = timestamp || new Date().toLocaleTimeString();
                messageDiv.innerHTML = content + `<span class="timestamp">${time}</span>`;
                
                chatArea.insertBefore(messageDiv, loading);
                chatArea.scrollTop = chatArea.scrollHeight;
                
                messageCount++;
                statMessages.textContent = messageCount;
                
                // Store in history
                chatHistory.push({ role, content, timestamp: time });
                exportBtn.disabled = false;
            }

            // ===== SUGGESTED QUESTIONS =====
            function displaySuggestedQuestions(questions) {
                suggestedQuestions.innerHTML = '';
                questions.forEach(q => {
                    const btn = document.createElement('button');
                    btn.className = 'suggested-btn';
                    btn.textContent = q.length > 50 ? q.slice(0, 50) + '...' : q;
                    btn.title = q;
                    btn.addEventListener('click', () => {
                        questionInput.value = q;
                        sendQuestion();
                    });
                    suggestedQuestions.appendChild(btn);
                });
                suggestedContainer.classList.add('active');
            }

            // ===== UPDATE DASHBOARD =====
            function updateDashboard(stats) {
                documentStats = stats;
                statPages.textContent = stats.pages || 0;
                statChunks.textContent = stats.chunks || 0;
                statSize.textContent = stats.size || '0 KB';
                dashboard.classList.add('active');
            }

            // ===== UPLOAD PDF =====
            fileInput.addEventListener('change', async function(e) {
                const file = this.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                status.className = 'status';
                status.textContent = '⏳ Uploading...';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        sessionId = data.session_id;
                        fileInfo.textContent = '📄 ' + file.name;
                        clearBtn.style.display = 'inline-block';
                        status.className = 'status success';
                        status.textContent = '✅ Loaded! Ask questions below.';
                        questionInput.disabled = false;
                        sendBtn.disabled = false;
                        
                        // Update dashboard
                        const fileSize = (file.size / 1024).toFixed(1) + ' KB';
                        updateDashboard({
                            pages: data.pages || 0,
                            chunks: data.chunks || 0,
                            size: fileSize
                        });
                        
                        // Reset chat
                        chatArea.innerHTML = '';
                        loading = document.createElement('div');
                        loading.className = 'loading';
                        loading.id = 'loading';
                        loading.innerHTML = '<div class="spinner"></div> Generating answer...';
                        chatArea.appendChild(loading);
                        messageCount = 0;
                        statMessages.textContent = 0;
                        chatHistory = [];
                        exportBtn.disabled = true;
                        
                        addMessage('assistant', '✅ PDF "' + file.name + '" loaded! Here are some questions you can ask:');
                        
                        // Get suggested questions
                        try {
                            const suggestResponse = await fetch('/suggest/' + sessionId);
                            const suggestData = await suggestResponse.json();
                            if (suggestData.success) {
                                displaySuggestedQuestions(suggestData.questions);
                            }
                        } catch (err) {
                            console.error('Error fetching suggestions:', err);
                        }
                    } else {
                        status.className = 'status error';
                        status.textContent = '❌ ' + data.message;
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = '❌ Upload failed: ' + error.message;
                }
            });

            // ===== CLEAR =====
            clearBtn.addEventListener('click', async function() {
                if (sessionId) {
                    try {
                        await fetch('/clear/' + sessionId, { method: 'DELETE' });
                    } catch (e) {}
                }
                sessionId = null;
                fileInfo.textContent = 'No file uploaded';
                clearBtn.style.display = 'none';
                status.className = 'status';
                status.textContent = '';
                questionInput.disabled = true;
                sendBtn.disabled = true;
                exportBtn.disabled = true;
                dashboard.classList.remove('active');
                suggestedContainer.classList.remove('active');
                messageCount = 0;
                statMessages.textContent = 0;
                chatHistory = [];
                chatArea.innerHTML = `
                    <div class="message assistant">
                        👋 Hello! Upload a PDF and I'll answer questions about it.
                    </div>
                    <div class="loading" id="loading">
                        <div class="spinner"></div> Generating answer...
                    </div>
                `;
                loading = document.getElementById('loading');
                fileInput.value = '';
            });

            // ===== SEND QUESTION =====
            async function sendQuestion() {
                const question = questionInput.value.trim();
                if (!question || !sessionId || isProcessing) return;
                
                isProcessing = true;
                addMessage('user', question);
                questionInput.value = '';
                questionInput.disabled = true;
                sendBtn.disabled = true;
                loading.classList.add('active');
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            question: question
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let message = data.answer;
                        if (data.sources && data.sources.length > 0) {
                            message += '<div class="sources">📚 Sources: ' + 
                                data.sources.map((s, i) => 'Page ' + s).join(', ') + 
                            '</div>';
                        }
                        addMessage('assistant', message);
                    } else {
                        addMessage('assistant', '❌ Error: ' + data.message);
                    }
                } catch (error) {
                    addMessage('assistant', '❌ Error: ' + error.message);
                } finally {
                    isProcessing = false;
                    loading.classList.remove('active');
                    questionInput.disabled = false;
                    sendBtn.disabled = false;
                    questionInput.focus();
                }
            }

            // ===== EXPORT CHAT =====
            exportBtn.addEventListener('click', async function() {
                if (!sessionId || chatHistory.length === 0) return;
                
                let text = '📄 PDF Chat Export\n';
                text += '='.repeat(50) + '\n\n';
                text += 'Exported: ' + new Date().toLocaleString() + '\n\n';
                
                chatHistory.forEach(msg => {
                    const label = msg.role === 'user' ? '👤 You' : '🤖 Assistant';
                    text += `[${msg.timestamp}] ${label}:\n${msg.content.replace(/<[^>]*>/g, '')}\n\n`;
                });
                
                // Download as file
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `chat_export_${new Date().toISOString().slice(0,10)}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            });

            // ===== EVENT LISTENERS =====
            sendBtn.addEventListener('click', sendQuestion);
            questionInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendQuestion();
            });
        </script>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        session_id = str(uuid.uuid4())
        temp_dir = f"./temp_{session_id}"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and process PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=f"./chroma_{session_id}"
        )
        
        sessions[session_id] = {
            "vector_store": vector_store,
            "file_name": file.filename,
            "temp_dir": temp_dir,
            "pages": len(documents),
            "chunks": len(chunks)
        }
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": f"Loaded {len(documents)} pages, {len(chunks)} chunks",
            "pages": len(documents),
            "chunks": len(chunks)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

@app.get("/suggest/{session_id}")
async def get_suggested_questions(session_id: str):
    """Get suggested questions for a document"""
    try:
        if session_id not in sessions:
            return JSONResponse({
                "success": False,
                "message": "Session not found"
            })
        
        # Get some context from the vector store
        vector_store = sessions[session_id]["vector_store"]
        # Get a sample of chunks
        all_docs = vector_store.get()
        if all_docs and all_docs.get('documents'):
            context = "\n".join(all_docs['documents'][:3])
        else:
            context = "Document about various topics."
        
        questions = generate_suggested_questions(context)
        return JSONResponse({
            "success": True,
            "questions": questions
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded PDF"""
    try:
        if request.session_id not in sessions:
            return JSONResponse({
                "success": False,
                "message": "Session not found. Please upload a PDF first."
            })
        
        vector_store = sessions[request.session_id]["vector_store"]
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(request.question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata.get('page', 'unknown') for doc in docs]
        
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context from the PDF:
{context}

Question: {request.question}

Answer the question based ONLY on the context above. If the answer is not in the context, say "I don't have enough information to answer this question."
"""
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = completion.choices[0].message.content
        
        return JSONResponse({
            "success": True,
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

@app.delete("/clear/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and clean up files"""
    if session_id in sessions:
        temp_dir = sessions[session_id].get("temp_dir")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        del sessions[session_id]
    return JSONResponse({"success": True})

@app.get("/export/{session_id}")
async def export_chat(session_id: str):
    """Export chat history"""
    # This is handled on the frontend
    return JSONResponse({"success": True})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)