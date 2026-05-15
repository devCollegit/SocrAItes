/* src/frontend/app.js */
document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const welcomeBanner = document.getElementById('welcome-banner');
    const currentPlan = document.getElementById('current-plan');
    const depthBtns = document.querySelectorAll('.depth-btn');
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
    const themeToggle = document.getElementById('theme-toggle');
    const statTurns = document.getElementById('stat-turns');
    const statDocs = document.getElementById('stat-docs');
    const docsList = document.getElementById('docs-list');
    const quickChips = document.querySelectorAll('.chip');
    const attachBtn = document.getElementById('attach-btn');
    const pdfUpload = document.getElementById('pdf-upload');

    // State
    let messages = [];
    let socraticDepth = 1;
    let sessionId = null;
    let isThinking = false;
    let turnCount = 0;
    let docCount = 0;

    // Markdown Configuration
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        sendBtn.disabled = !this.value.trim();
        document.getElementById('char-count').innerText = this.value.length || '';
    });

    // Sidebar Toggle
    const toggleSidebar = () => {
        sidebar.classList.toggle('collapsed');
        sidebarOpenBtn.style.display = sidebar.classList.contains('collapsed') ? 'flex' : 'none';
    };
    sidebarToggle.addEventListener('click', toggleSidebar);
    sidebarOpenBtn.addEventListener('click', toggleSidebar);

    // Theme Toggle
    themeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', nextTheme);
        document.getElementById('theme-icon-dark').style.display = nextTheme === 'dark' ? 'block' : 'none';
        document.getElementById('theme-icon-light').style.display = nextTheme === 'light' ? 'block' : 'none';
    });

    // Depth Selection
    depthBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            depthBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            socraticDepth = parseInt(btn.dataset.depth);
        });
    });

    // Quick Start Chips
    quickChips.forEach(chip => {
        chip.addEventListener('click', () => {
            userInput.value = chip.dataset.msg;
            userInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    });

    // Send Message
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text || isThinking) return;

        if (welcomeBanner) welcomeBanner.style.display = 'none';

        // UI Update
        addMessage(text, 'user');
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
        
        isThinking = true;
        const loadingId = addLoadingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [...messages, { role: 'user', content: text }],
                    socratic_depth: socraticDepth,
                    session_id: sessionId
                })
            });

            if (!response.ok) throw new Error('API request failed');

            const data = await response.json();
            
            // Update State
            sessionId = data.session_id;
            messages.push({ role: 'user', content: text });
            messages.push({ role: 'assistant', content: data.answer });
            turnCount++;
            statTurns.innerText = turnCount;

            // Remove Loading & Add AI Response
            removeLoadingIndicator(loadingId);
            addMessage(data.answer, 'ai');

            // Update Plan
            if (data.plan) {
                currentPlan.innerHTML = `<div class="plan-content">${marked.parse(data.plan)}</div>`;
                currentPlan.classList.add('updated');
                setTimeout(() => currentPlan.classList.remove('updated'), 1000);
            }

            // Update Retrieved Docs
            if (data.retrieved_docs && data.retrieved_docs.length > 0) {
                docCount = data.retrieved_docs.length;
                statDocs.innerText = docCount;
                docsList.innerHTML = data.retrieved_docs.map(doc => `
                    <div class="doc-item">
                        <svg viewBox="0 0 20 20" fill="currentColor"><path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v4a2 2 0 01-2 2H4a2 2 0 01-2-2v-4z"/></svg>
                        <span>${doc.metadata?.source || '강의 자료 일부'}</span>
                    </div>
                `).join('');
            }

        } catch (error) {
            console.error(error);
            removeLoadingIndicator(loadingId);
            addMessage('죄송합니다. 서버와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'ai');
        } finally {
            isThinking = false;
        }
    }

    function addMessage(text, role) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'msg-avatar';
        avatarDiv.innerText = role === 'ai' ? 'S' : (role === 'system' ? '⚙️' : 'U');

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role === 'system') {
            contentDiv.innerHTML = `<span class="system-text">${marked.parse(text)}</span>`;
        } else {
            contentDiv.innerHTML = role === 'ai' ? marked.parse(text) : text.replace(/\n/g, '<br>');
        }
        
        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai';
        msgDiv.id = id;
        msgDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeLoadingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTo({
            top: chatMessages.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // PDF Upload Handling
    if (attachBtn && pdfUpload) {
        attachBtn.addEventListener('click', () => pdfUpload.click());
        
        pdfUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            // Show system message for upload starting
            addMessage(`📄 **${file.filename || file.name}** 파일을 업로드 중입니다...`, 'system');
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Upload failed');
                }
                
                const data = await response.json();
                addMessage(`✅ **${data.filename}** 등록 완료! (${data.chunks_added}개의 지식 조각 추출)`, 'system');
                
                // Update doc count (dummy update since we don't know total docs in store)
                // In a real app, you might fetch the current count or list
            } catch (error) {
                console.error(error);
                addMessage(`❌ 업로드 실패: ${error.message}`, 'system');
            } finally {
                // Clear input so same file can be uploaded again if needed
                pdfUpload.value = '';
            }
        });
    }

    newChatBtn.addEventListener('click', () => {
        if (confirm('대화 내용을 초기화하고 새로 시작하시겠습니까?')) {
            chatMessages.innerHTML = '';
            welcomeBanner.style.display = 'flex';
            messages = [];
            sessionId = null;
            turnCount = 0;
            docCount = 0;
            statTurns.innerText = '0';
            statDocs.innerText = '0';
            currentPlan.innerHTML = '<div class="plan-empty"><p>질문하면 AI가<br>학습 계획을 세웁니다</p></div>';
            docsList.innerHTML = '<div class="docs-empty">참조 자료 없음</div>';
        }
    });
});
