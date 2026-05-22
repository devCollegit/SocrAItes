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
    const registeredDocsList = document.getElementById('registered-docs-list');

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

            // Read the SSE response stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let finalResult = null;

            const processLine = (line) => {
                if (!line.startsWith('data: ')) return;
                const jsonStr = line.slice(6).trim();
                if (!jsonStr) return;
                try {
                    const data = JSON.parse(jsonStr);
                    if (data.type === 'node_end') {
                        updateProgressStep(loadingId, data.node, data.output);
                    } else if (data.type === 'final_result') {
                        finalResult = data;
                    } else if (data.type === 'error') {
                        throw new Error(data.detail);
                    }
                } catch (e) {
                    console.error('Error parsing stream chunk:', e);
                }
            };

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    // flush remaining buffer content before exiting
                    buffer += decoder.decode();
                    buffer.split('\n').forEach(line => processLine(line.trim()));
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                lines.forEach(line => processLine(line));
            }

            if (!finalResult) throw new Error('No final result returned from stream');
            const data = finalResult;

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
                docsList.innerHTML = data.retrieved_docs.map((doc, idx) => {
                    const sourceName = doc.metadata?.source || '강의 자료 일부';
                    const pageNum = doc.metadata?.page || 1;
                    const text = doc.text || '';
                    return `
                        <div class="doc-item-collapsible" data-index="${idx}">
                            <div class="doc-item-header">
                                <div class="doc-item-left">
                                    <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/></svg>
                                    <span class="doc-title-text" title="${sourceName}">${sourceName}</span>
                                    <span class="doc-page-badge">p.${pageNum}</span>
                                </div>
                                <span class="doc-expand-icon">▼</span>
                            </div>
                            <div class="doc-snippet-content">${text}</div>
                        </div>
                    `;
                }).join('');

                // Add toggle click handlers
                docsList.querySelectorAll('.doc-item-header').forEach(header => {
                    header.addEventListener('click', () => {
                        const item = header.parentElement;
                        item.classList.toggle('active');
                    });
                });
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
            <div class="message-content node-progress-card">
                <div class="progress-title">⚙️ AI 에이전트 분석 및 추론 중...</div>
                <div class="node-steps" id="${id}-steps"></div>
                <div class="typing-indicator" id="${id}-typing">
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

    const nodeLabels = {
        'query_contextualizer': '🔍 질문 분석 및 문맥 이해',
        'coordinator': '🧭 질문 유형 분류 및 탐구 결정',
        'planner': '📋 Socratic 학습 계획 수립',
        'retrieval': '📚 Elasticsearch 강의 자료 검색',
        'supervisor': '🧠 Socratic 튜터 답변 생성',
        'evaluator': '⚖️ 답변 품질 검증 및 자가 교정',
        'direct_response': '💬 일반 안내 및 대화 답변 작성'
    };

    function updateProgressStep(loadingId, node, output) {
        const stepsContainer = document.getElementById(`${loadingId}-steps`);
        if (!stepsContainer) return;

        let detail = '';
        if (node === 'query_contextualizer') {
            detail = `재구성 결과: "${output.contextualized_query}"`;
        } else if (node === 'coordinator') {
            detail = `판별 결과: ${output.next_step === 'planner' ? '개념 학습 (PLAN)' : '일반 대화 (DIRECT)'}`;
        } else if (node === 'planner') {
            detail = `Socratic 튜터링 가이드 구성 완료`;
        } else if (node === 'retrieval') {
            const count = output.retrieved_docs ? output.retrieved_docs.length : 0;
            detail = `검색 완료 (${count}개 조각 참조)`;
        } else if (node === 'supervisor') {
            detail = `소크라테스식 응답 가이드라인 작성 완료`;
        } else if (node === 'evaluator') {
            detail = `품질 기준 검사 통과`;
        } else if (node === 'direct_response') {
            detail = `일반 대화형 응답 생성 완료`;
        }

        const label = nodeLabels[node] || node;
        const stepDiv = document.createElement('div');
        stepDiv.className = 'node-step';
        stepDiv.innerHTML = `
            <div class="node-step-header">
                <span class="node-step-icon">✅</span>
                <span class="node-step-label">${label}</span>
            </div>
            ${detail ? `<div class="node-step-detail">${detail}</div>` : ''}
        `;
        stepsContainer.appendChild(stepDiv);
        scrollToBottom();
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
                if (data.status === 'duplicate') {
                    addMessage(`⚠️ **${data.filename}** 은 이미 등록된 파일입니다.`, 'system');
                } else {
                    addMessage(`✅ **${data.filename}** 등록 완료! (${data.chunks_added}개의 지식 조각 추출)`, 'system');
                }
                
                // Update registered documents list
                await fetchRegisteredDocuments();

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

    // Fetch and render registered PDF documents
    async function fetchRegisteredDocuments() {
        try {
            const response = await fetch('/documents');
            if (!response.ok) throw new Error('Failed to fetch documents');
            const data = await response.json();
            
            if (data.documents && data.documents.length > 0) {
                registeredDocsList.innerHTML = data.documents.map(filename => `
                    <div class="doc-item" data-filename="${filename}">
                        <div class="doc-item-left">
                            <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/></svg>
                            <span title="${filename}">${filename}</span>
                        </div>
                        <button class="doc-delete-btn" title="삭제" data-filename="${filename}">
                            <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
                        </button>
                    </div>
                `).join('');
                
                // Add delete event listeners
                registeredDocsList.querySelectorAll('.doc-delete-btn').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const filename = btn.dataset.filename;
                        if (confirm(`"${filename}" 문서를 데이터베이스에서 삭제하시겠습니까? 관련 모든 지식이 삭제됩니다.`)) {
                            await deleteRegisteredDocument(filename);
                        }
                    });
                });
            } else {
                registeredDocsList.innerHTML = '<div class="docs-empty">등록된 문서 없음</div>';
            }
        } catch (error) {
            console.error('Error fetching registered documents:', error);
            registeredDocsList.innerHTML = '<div class="docs-empty">목록 로드 실패</div>';
        }
    }

    // Delete a registered document
    async function deleteRegisteredDocument(filename) {
        addMessage(`⚙️ **${filename}** 삭제를 진행 중입니다...`, 'system');
        try {
            const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Delete failed');
            }
            const data = await response.json();
            if (data.status === 'success') {
                addMessage(`🗑️ **${filename}** 삭제 완료! (총 ${data.deleted_chunks}개의 지식 조각이 제거되었습니다.)`, 'system');
            } else {
                addMessage(`⚠️ **${filename}** 을 찾을 수 없거나 이미 삭제되었습니다.`, 'system');
            }
            await fetchRegisteredDocuments();
        } catch (error) {
            console.error('Error deleting document:', error);
            addMessage(`❌ 삭제 실패: ${error.message}`, 'system');
        }
    }

    // Initial fetch of registered documents
    fetchRegisteredDocuments();
});

