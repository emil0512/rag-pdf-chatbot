import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import bcrypt

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===== DATABASE =====
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        pages INTEGER DEFAULT 0,
        chunks INTEGER DEFAULT 0,
        size_kb REAL DEFAULT 0,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )''')
    conn.commit()
    conn.close()

init_db()

def get_user_by_email(email):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(email, password_hash):
    user_id = str(uuid.uuid4())
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
              (user_id, email, password_hash))
    conn.commit()
    conn.close()
    return user_id

def get_user_documents(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,))
    docs = c.fetchall()
    conn.close()
    return docs

def save_document(user_id, file_name, file_path, pages, chunks, size_kb):
    doc_id = str(uuid.uuid4())
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO documents (id, user_id, file_name, file_path, pages, chunks, size_kb) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (doc_id, user_id, file_name, file_path, pages, chunks, size_kb))
    conn.commit()
    conn.close()
    return doc_id

def delete_document(doc_id, user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT file_path FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    result = c.fetchone()
    if result:
        file_path = result[0]
        if os.path.exists(file_path):
            try:
                shutil.rmtree(os.path.dirname(file_path))
            except:
                pass
        c.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
        c.execute("DELETE FROM chat_history WHERE document_id = ?", (doc_id,))
        conn.commit()
    conn.close()
    return True

def save_chat_message(doc_id, role, content):
    msg_id = str(uuid.uuid4())
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (id, document_id, role, content) VALUES (?, ?, ?, ?)",
              (msg_id, doc_id, role, content))
    conn.commit()
    conn.close()

def get_chat_history(doc_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_history WHERE document_id = ? ORDER BY timestamp ASC", (doc_id,))
    history = c.fetchall()
    conn.close()
    return history

def hash_password(password):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

sessions = {}

def create_session(user_id):
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"user_id": user_id, "created_at": datetime.now()}
    return session_id

def get_session_user(session_id):
    if session_id in sessions:
        return sessions[session_id]["user_id"]
    return None

def clear_session(session_id):
    if session_id in sessions:
        del sessions[session_id]

# ===== LOGIN PAGE =====
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PDF Chatbot - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .auth-container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 420px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .auth-container h1 { font-size: 28px; color: #2d3748; margin-bottom: 8px; }
        .auth-container .subtitle { color: #718096; margin-bottom: 30px; font-size: 14px; }
        .form-group { margin-bottom: 20px; }
        .form-group label { display: block; font-weight: 600; color: #4a5568; margin-bottom: 6px; font-size: 14px; }
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 14px;
            transition: border 0.3s;
            outline: none;
        }
        .form-group input:focus { border-color: #667eea; }
        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-primary:hover { background: #5a67d8; transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        .switch-link { text-align: center; margin-top: 20px; color: #718096; font-size: 14px; }
        .switch-link a { color: #667eea; text-decoration: none; font-weight: 600; }
        .switch-link a:hover { text-decoration: underline; }
        .error-msg { background: #fed7d7; color: #c53030; padding: 10px 14px; border-radius: 8px; font-size: 14px; margin-bottom: 16px; display: none; }
        .success-msg { background: #c6f6d5; color: #276749; padding: 10px 14px; border-radius: 8px; font-size: 14px; margin-bottom: 16px; display: none; }
    </style>
</head>
<body>
    <div class="auth-container" id="app">
        <h1 id="formTitle">Welcome Back</h1>
        <p class="subtitle" id="formSubtitle">Login to chat with your PDFs</p>
        <div class="error-msg" id="errorMsg"></div>
        <div class="success-msg" id="successMsg"></div>
        <form id="authForm">
            <div class="form-group">
                <label>Email</label>
                <input type="email" id="email" required placeholder="you@example.com">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="password" required placeholder="password">
            </div>
            <button type="submit" class="btn btn-primary" id="submitBtn">Login</button>
        </form>
        <div class="switch-link">
            <span id="switchText">Don't have an account?</span>
            <a href="#" id="switchLink">Sign Up</a>
        </div>
    </div>
    <script>
        let isLogin = true;
        const form = document.getElementById('authForm');
        const email = document.getElementById('email');
        const password = document.getElementById('password');
        const submitBtn = document.getElementById('submitBtn');
        const formTitle = document.getElementById('formTitle');
        const formSubtitle = document.getElementById('formSubtitle');
        const switchText = document.getElementById('switchText');
        const switchLink = document.getElementById('switchLink');
        const errorMsg = document.getElementById('errorMsg');
        const successMsg = document.getElementById('successMsg');

        function showError(msg) {
            errorMsg.textContent = msg;
            errorMsg.style.display = 'block';
            successMsg.style.display = 'none';
        }
        function showSuccess(msg) {
            successMsg.textContent = msg;
            successMsg.style.display = 'block';
            errorMsg.style.display = 'none';
        }
        function hideMessages() {
            errorMsg.style.display = 'none';
            successMsg.style.display = 'none';
        }

        switchLink.addEventListener('click', function(e) {
            e.preventDefault();
            isLogin = !isLogin;
            if (isLogin) {
                formTitle.textContent = 'Welcome Back';
                formSubtitle.textContent = 'Login to chat with your PDFs';
                submitBtn.textContent = 'Login';
                switchText.textContent = "Don't have an account?";
                switchLink.textContent = 'Sign Up';
            } else {
                formTitle.textContent = 'Create Account';
                formSubtitle.textContent = 'Start chatting with your PDFs for free';
                submitBtn.textContent = 'Sign Up';
                switchText.textContent = 'Already have an account?';
                switchLink.textContent = 'Login';
            }
            hideMessages();
        });

        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            hideMessages();
            const endpoint = isLogin ? '/login' : '/signup';
            const data = { email: email.value.trim(), password: password.value };
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (result.success) {
                    if (isLogin) {
                        window.location.href = '/dashboard';
                    } else {
                        showSuccess('Account created! Please login.');
                        isLogin = true;
                        formTitle.textContent = 'Welcome Back';
                        formSubtitle.textContent = 'Login to chat with your PDFs';
                        submitBtn.textContent = 'Login';
                        switchText.textContent = "Don't have an account?";
                        switchLink.textContent = 'Sign Up';
                        password.value = '';
                    }
                } else {
                    showError(result.message);
                }
            } catch (err) {
                showError('Network error. Please try again.');
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get_login():
    return HTMLResponse(LOGIN_HTML)

@app.post("/signup")
async def signup(request: Request):
    data = await request.json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return JSONResponse({"success": False, "message": "Email and password required"})
    
    existing = get_user_by_email(email)
    if existing:
        return JSONResponse({"success": False, "message": "Email already registered"})
    
    password_hash = hash_password(password)
    user_id = create_user(email, password_hash)
    
    return JSONResponse({"success": True, "message": "Account created successfully"})

@app.post("/login")
async def login(request: Request):
    data = await request.json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return JSONResponse({"success": False, "message": "Email and password required"})
    
    user = get_user_by_email(email)
    if not user:
        return JSONResponse({"success": False, "message": "Invalid credentials"})
    
    if not verify_password(password, user[2]):
        return JSONResponse({"success": False, "message": "Invalid credentials"})
    
    session_id = create_session(user[0])
    response = JSONResponse({"success": True})
    response.set_cookie(key="session_id", value=session_id, httponly=True, max_age=86400)
    return response

# ===== DASHBOARD =====
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PDF Chatbot - Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; min-height: 100vh; }}
        .navbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 15px 30px; display: flex;
            justify-content: space-between; align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .navbar h1 {{ font-size: 22px; }}
        .navbar .user-info {{ display: flex; align-items: center; gap: 15px; }}
        .navbar .email {{ font-size: 14px; opacity: 0.9; }}
        .btn-logout {{
            background: rgba(255,255,255,0.2); border: none; color: white;
            padding: 8px 16px; border-radius: 8px; cursor: pointer;
            font-weight: 600; transition: background 0.3s;
        }}
        .btn-logout:hover {{ background: rgba(255,255,255,0.3); }}
        .container {{ max-width: 1200px; margin: 30px auto; padding: 0 20px; }}
        .upload-section {{
            background: white; border-radius: 16px; padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 30px;
        }}
        .upload-section h2 {{ margin-bottom: 16px; color: #2d3748; }}
        .upload-area {{
            border: 2px dashed #e2e8f0; border-radius: 12px; padding: 40px;
            text-align: center; transition: border 0.3s; cursor: pointer;
        }}
        .upload-area:hover {{ border-color: #667eea; }}
        .upload-area input {{ display: none; }}
        .upload-area .icon {{ font-size: 48px; margin-bottom: 10px; }}
        .upload-area .text {{ color: #718096; }}
        .upload-area .highlight {{ color: #667eea; font-weight: 600; }}
        .documents-grid {{
            display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }}
        .doc-card {{
            background: white; border-radius: 16px; padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }}
        .doc-card:hover {{ transform: translateY(-4px); box-shadow: 0 4px 20px rgba(0,0,0,0.12); }}
        .doc-card .doc-name {{ font-weight: 600; color: #2d3748; font-size: 16px; }}
        .doc-card .doc-meta {{ color: #718096; font-size: 13px; margin-top: 4px; }}
        .doc-card .doc-actions {{ margin-top: 12px; display: flex; gap: 8px; }}
        .doc-card .doc-actions .btn {{
            padding: 6px 14px; border: none; border-radius: 6px; cursor: pointer;
            font-size: 13px; font-weight: 600; transition: all 0.2s;
            text-decoration: none; display: inline-block;
        }}
        .btn-chat {{ background: #667eea; color: white; }}
        .btn-chat:hover {{ background: #5a67d8; }}
        .btn-delete {{ background: #fc8181; color: white; }}
        .btn-delete:hover {{ background: #f56565; }}
        .empty-state {{ text-align: center; padding: 60px 20px; color: #718096; }}
        .empty-state .icon {{ font-size: 64px; margin-bottom: 16px; }}
        .status {{ margin-top: 12px; font-size: 14px; display: none; }}
        .status.success {{ color: #48bb78; display: block; }}
        .status.error {{ color: #fc8181; display: block; }}
        .status.loading {{ color: #667eea; display: block; }}
        @media (max-width: 768px) {{
            .navbar {{ padding: 12px 16px; flex-wrap: wrap; gap: 8px; }}
            .navbar h1 {{ font-size: 18px; }}
            .documents-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <h1>PDF Chatbot</h1>
        <div class="user-info">
            <span class="email">{email}</span>
            <form action="/logout" method="post">
                <button type="submit" class="btn-logout">Logout</button>
            </form>
        </div>
    </div>
    <div class="container">
        <div class="upload-section">
            <h2>Upload a PDF</h2>
            <div class="upload-area" id="uploadArea">
                <div class="icon">📁</div>
                <div class="text">Drag & drop your PDF here or <span class="highlight">browse</span></div>
                <input type="file" id="fileInput" accept=".pdf">
                <div class="status" id="uploadStatus"></div>
            </div>
        </div>
        <div class="documents-grid" id="documentsGrid">
            {docs}
        </div>
        {empty}
    </div>
    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const uploadStatus = document.getElementById('uploadStatus');

        uploadArea.addEventListener('click', function() {{ fileInput.click(); }});
        uploadArea.addEventListener('dragover', function(e) {{
            e.preventDefault();
            uploadArea.style.borderColor = '#667eea';
        }});
        uploadArea.addEventListener('dragleave', function() {{
            uploadArea.style.borderColor = '#e2e8f0';
        }});
        uploadArea.addEventListener('drop', function(e) {{
            e.preventDefault();
            uploadArea.style.borderColor = '#e2e8f0';
            if (e.dataTransfer.files.length) {{
                fileInput.files = e.dataTransfer.files;
                uploadFile(e.dataTransfer.files[0]);
            }}
        }});

        fileInput.addEventListener('change', function() {{
            if (fileInput.files.length) {{
                uploadFile(fileInput.files[0]);
            }}
        }});

        async function uploadFile(file) {{
            const formData = new FormData();
            formData.append('file', file);
            uploadStatus.className = 'status loading';
            uploadStatus.textContent = 'Uploading and processing...';
            try {{
                const response = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await response.json();
                if (data.success) {{
                    uploadStatus.className = 'status success';
                    uploadStatus.textContent = 'Uploaded successfully!';
                    setTimeout(function() {{ location.reload(); }}, 1000);
                }} else {{
                    uploadStatus.className = 'status error';
                    uploadStatus.textContent = data.message;
                }}
            }} catch (err) {{
                uploadStatus.className = 'status error';
                uploadStatus.textContent = 'Upload failed';
            }}
        }}

        async function deleteDoc(docId) {{
            if (!confirm('Delete this document?')) return;
            try {{
                const response = await fetch('/document/' + docId, {{
                    method: 'DELETE'
                }});
                const data = await response.json();
                if (data.success) {{
                    location.reload();
                }}
            }} catch (err) {{
                alert('Error deleting document');
            }}
        }}
    </script>
</body>
</html>
"""
@app.get("/dashboard")
async def get_dashboard(request: Request):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    
    user = get_user_by_id(user_id)
    if not user:
        return RedirectResponse(url="/", status_code=303)
    
    docs = get_user_documents(user_id)
    
    docs_html = ""
    for doc in docs:
        docs_html += f'''
        <div class="doc-card" data-id="{doc[0]}">
            <div class="doc-name">📄 {doc[2]}</div>
            <div class="doc-meta">
                {doc[4]} pages · {doc[5]} chunks · {doc[6]:.1f} KB
                <br>Uploaded: {doc[7]}
            </div>
            <div class="doc-actions">
                <a href="/chat/{doc[0]}" class="btn btn-chat">💬 Chat</a>
                <button class="btn btn-delete" onclick="deleteDoc('{doc[0]}')">🗑️ Delete</button>
            </div>
        </div>
        '''
    
    empty_html = '' if docs else '''
    <div class="empty-state">
        <div class="icon">📭</div>
        <h3>No documents yet</h3>
        <p>Upload your first PDF to get started!</p>
    </div>
    '''
    
    return HTMLResponse(DASHBOARD_HTML.format(email=user[1], docs=docs_html, empty=empty_html))

# ===== CHAT PAGE =====
@app.get("/chat/{doc_id}")
async def get_chat(request: Request, doc_id: str):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return RedirectResponse(url="/", status_code=303)
    
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
    doc = c.fetchone()
    conn.close()
    
    if not doc:
        return HTMLResponse("Document not found", status_code=404)
    
    history = get_chat_history(doc_id)
    
    history_html = ""
    for msg in history:
        history_html += f'''
        <div class="message {msg[0]}">
            {msg[1]}
            <span class="timestamp">{msg[2]}</span>
        </div>
        '''
    
    if not history:
        history_html = '<div class="message assistant">👋 Ask me anything about this document!</div>'
    
    chat_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Chat - {doc[2]}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .navbar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        .navbar-left {{ display: flex; align-items: center; gap: 15px; }}
        .navbar h1 {{ font-size: 18px; }}
        .navbar .back-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            text-decoration: none;
        }}
        .navbar .back-btn:hover {{ background: rgba(255,255,255,0.3); }}
        .chat-container {{
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 20px;
        }}
        .chat-area {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            min-height: 400px;
            max-height: 600px;
        }}
        .message {{
            margin-bottom: 12px;
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 12px;
            line-height: 1.6;
            word-wrap: break-word;
        }}
        .message.user {{
            background: #667eea;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }}
        .message.assistant {{
            background: #f7fafc;
            color: #2d3748;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
        }}
        .message .timestamp {{
            font-size: 10px;
            color: #a0aec0;
            margin-top: 4px;
            display: block;
        }}
        .message.user .timestamp {{ color: rgba(255,255,255,0.7); }}
        .message .sources {{
            margin-top: 8px;
            font-size: 12px;
            color: #718096;
            border-top: 1px solid #e2e8f0;
            padding-top: 8px;
        }}
        .input-area {{
            display: flex;
            gap: 10px;
            padding: 15px 0;
            background: transparent;
        }}
        .input-area input {{
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 14px;
            outline: none;
            transition: border 0.3s;
        }}
        .input-area input:focus {{ border-color: #667eea; }}
        .input-area input:disabled {{ opacity: 0.6; cursor: not-allowed; }}
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }}
        .btn-success {{ background: #48bb78; color: white; }}
        .btn-success:hover {{ background: #38a169; transform: translateY(-2px); }}
        .btn-success:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .loading {{ display: none; text-align: center; padding: 15px; color: #718096; }}
        .loading.active {{ display: block; }}
        .spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #e2e8f0;
            border-radius: 50%;
            border-top-color: #667eea;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .suggested-container {{
            padding: 10px 0;
            display: none;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .suggested-container.active {{ display: flex; }}
        .suggested-label {{
            font-size: 12px;
            color: #718096;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            margin-right: 8px;
        }}
        .suggested-btn {{
            padding: 4px 14px;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            background: white;
            color: #4a5568;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .suggested-btn:hover {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}
        .export-btn {{
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }}
        .export-btn:hover {{ background: rgba(255,255,255,0.3); }}
        .export-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
        @media (max-width: 768px) {{
            .chat-container {{ padding: 10px; }}
            .chat-area {{ min-height: 300px; max-height: 450px; }}
            .message {{ max-width: 90%; }}
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="navbar-left">
            <a href="/dashboard" class="back-btn">← Back</a>
            <h1>📄 {doc[2]}</h1>
        </div>
        <div>
            <button class="export-btn" id="exportBtn" disabled>📥 Export Chat</button>
        </div>
    </div>
    <div class="chat-container">
        <div class="chat-area" id="chatArea">
            {history_html}
            <div class="loading" id="loading">
                <div class="spinner"></div> Generating answer...
            </div>
        </div>
        <div class="suggested-container" id="suggestedContainer">
            <span class="suggested-label">💡 Try:</span>
            <div id="suggestedQuestions"></div>
        </div>
        <div class="input-area">
            <input type="text" id="questionInput" placeholder="Ask a question...">
            <button class="btn btn-success" id="sendBtn">Send</button>
        </div>
    </div>
    <script>
        const docId = '{doc_id}';
        const chatArea = document.getElementById('chatArea');
        const questionInput = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');
        const exportBtn = document.getElementById('exportBtn');
        const suggestedContainer = document.getElementById('suggestedContainer');
        const suggestedQuestions = document.getElementById('suggestedQuestions');
        let isProcessing = false;
        let messageCount = {len(history)};

        async function loadSuggestions() {{
            try {{
                const response = await fetch('/suggest/' + docId);
                const data = await response.json();
                if (data.success && data.questions) {{
                    suggestedQuestions.innerHTML = '';
                    data.questions.forEach(function(q) {{
                        const btn = document.createElement('button');
                        btn.className = 'suggested-btn';
                        btn.textContent = q.length > 50 ? q.slice(0, 50) + '...' : q;
                        btn.title = q;
                        btn.addEventListener('click', function() {{
                            questionInput.value = q;
                            sendQuestion();
                        }});
                        suggestedQuestions.appendChild(btn);
                    }});
                    suggestedContainer.classList.add('active');
                }}
            }} catch (err) {{
                console.error('Error loading suggestions:', err);
            }}
        }}

        if ({len(history)} === 0) {{
            loadSuggestions();
        }}

        function addMessage(role, content, timestamp) {{
            const div = document.createElement('div');
            div.className = 'message ' + role;
            const time = timestamp || new Date().toLocaleTimeString();
            div.innerHTML = content + '<span class="timestamp">' + time + '</span>';
            chatArea.insertBefore(div, loading);
            chatArea.scrollTop = chatArea.scrollHeight;
            messageCount++;
            exportBtn.disabled = false;
            suggestedContainer.classList.remove('active');
        }}

        async function sendQuestion() {{
            const question = questionInput.value.trim();
            if (!question || isProcessing) return;
            isProcessing = true;
            addMessage('user', question);
            questionInput.value = '';
            questionInput.disabled = true;
            sendBtn.disabled = true;
            loading.classList.add('active');
            try {{
                const response = await fetch('/ask_doc', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ doc_id: docId, question: question }})
                }});
                const data = await response.json();
                if (data.success) {{
                    let message = data.answer;
                    if (data.sources && data.sources.length > 0) {{
                        message += '<div class="sources">📚 Sources: ' +
                            data.sources.map(function(s, i) {{ return 'Page ' + s; }}).join(', ') +
                        '</div>';
                    }}
                    addMessage('assistant', message);
                }} else {{
                    addMessage('assistant', 'Error: ' + data.message);
                }}
            }} catch (err) {{
                addMessage('assistant', 'Error: ' + err.message);
            }} finally {{
                isProcessing = false;
                loading.classList.remove('active');
                questionInput.disabled = false;
                sendBtn.disabled = false;
                questionInput.focus();
            }}
        }}

        exportBtn.addEventListener('click', async function() {{
            try {{
                const response = await fetch('/export/' + docId);
                const data = await response.json();
                if (data.success) {{
                    let text = 'Chat Export - {doc[2]}\\n';
                    text += '='.repeat(50) + '\\n\\n';
                    text += 'Exported: ' + new Date().toLocaleString() + '\\n\\n';
                    data.history.forEach(function(msg) {{
                        const label = msg.role === 'user' ? 'You' : 'Assistant';
                        text += '[' + msg.timestamp + '] ' + label + ':\\n' + msg.content.replace(/<[^>]*>/g, '') + '\\n\\n';
                    }});
                    const blob = new Blob([text], {{ type: 'text/plain' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'chat_' + new Date().toISOString().slice(0,10) + '.txt';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}
            }} catch (err) {{
                alert('Error exporting chat');
            }}
        }});

        sendBtn.addEventListener('click', sendQuestion);
        questionInput.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') sendQuestion();
        }});
    </script>
</body>
</html>
'''
    return HTMLResponse(chat_html)

# ===== API ROUTES =====
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return JSONResponse({"success": False, "message": "Not authenticated"})
    
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse({"success": False, "message": "Only PDF files are allowed"})
        
        user_folder = f"./user_data/{user_id}"
        os.makedirs(user_folder, exist_ok=True)
        
        file_path = os.path.join(user_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        vector_path = f"./vectors/{user_id}/{file.filename.replace('.pdf', '')}"
        os.makedirs(vector_path, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_path
        )
        
        file_size = os.path.getsize(file_path) / 1024
        doc_id = save_document(user_id, file.filename, file_path, len(documents), len(chunks), file_size)
        
        return JSONResponse({
            "success": True,
            "doc_id": doc_id,
            "pages": len(documents),
            "chunks": len(chunks)
        })
        
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

@app.post("/ask_doc")
async def ask_doc(request: Request):
    data = await request.json()
    doc_id = data.get('doc_id')
    question = data.get('question')
    
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return JSONResponse({"success": False, "message": "Not authenticated"})
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT file_path FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
        doc = c.fetchone()
        conn.close()
        
        if not doc:
            return JSONResponse({"success": False, "message": "Document not found"})
        
        vector_path = f"./vectors/{user_id}/{os.path.basename(doc[0]).replace('.pdf', '')}"
        vector_store = Chroma(persist_directory=vector_path, embedding_function=embeddings)
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        
        context = "\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get('page', 'unknown') for d in docs]
        
        prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context from the PDF:
{context}

Question: {question}

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
        
        save_chat_message(doc_id, "user", question)
        save_chat_message(doc_id, "assistant", answer)
        
        return JSONResponse({
            "success": True,
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

@app.get("/suggest/{doc_id}")
async def get_suggestions(request: Request, doc_id: str):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return JSONResponse({"success": False, "message": "Not authenticated"})
    
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT file_path FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))
        doc = c.fetchone()
        conn.close()
        
        if not doc:
            return JSONResponse({"success": False, "message": "Document not found"})
        
        vector_path = f"./vectors/{user_id}/{os.path.basename(doc[0]).replace('.pdf', '')}"
        vector_store = Chroma(persist_directory=vector_path, embedding_function=embeddings)
        
        all_docs = vector_store.get()
        context = "\n".join(all_docs['documents'][:3]) if all_docs.get('documents') else "Document about various topics."
        
        prompt = f"""Based on this document context, generate 5 questions a user might ask.
Context preview:
{context[:2000]}

Return only the questions, one per line, numbered 1-5."""
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Generate questions about documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        questions = []
        for line in completion.choices[0].message.content.strip().split('\n'):
            line = line.strip()
            if line and any(c.isdigit() for c in line[:3]):
                parts = line.split('.', 1)
                if len(parts) > 1:
                    questions.append(parts[1].strip())
                else:
                    parts = line.split(')', 1)
                    if len(parts) > 1:
                        questions.append(parts[1].strip())
            elif line:
                questions.append(line)
        
        while len(questions) < 5:
            questions.append("What are the key points in this document?")
        
        return JSONResponse({"success": True, "questions": questions[:5]})
        
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

@app.get("/export/{doc_id}")
async def export_chat(request: Request, doc_id: str):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return JSONResponse({"success": False, "message": "Not authenticated"})
    
    history = get_chat_history(doc_id)
    return JSONResponse({
        "success": True,
        "history": [{"role": h[0], "content": h[1], "timestamp": h[2]} for h in history]
    })

@app.delete("/document/{doc_id}")
async def delete_doc(request: Request, doc_id: str):
    session_id = request.cookies.get("session_id")
    user_id = get_session_user(session_id)
    
    if not user_id:
        return JSONResponse({"success": False, "message": "Not authenticated"})
    
    delete_document(doc_id, user_id)
    return JSONResponse({"success": True})

@app.post("/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    clear_session(session_id)
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("session_id")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)