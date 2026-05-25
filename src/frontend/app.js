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
    const statFrustration = document.getElementById('stat-frustration');
    const quickChips = document.querySelectorAll('.chip');
    const attachBtn = document.getElementById('attach-btn');
    const pdfUpload = document.getElementById('pdf-upload');
    const registeredDocsList = document.getElementById('registered-docs-list');
    const sessionList = document.getElementById('session-list');
    const scheduleList = document.getElementById('schedule-list');

    // State
    let messages = [];
    let socraticDepth = 1;
    let sessionId = null;
    let isThinking = false;
    let turnCount = 0;
    let docCount = 0;
    let frustrationLevel = 0;

    // 커스텀 확인 모달
    const deleteModal = document.getElementById('delete-modal');
    function showDeleteModal() {
        return new Promise(resolve => {
            deleteModal.style.display = 'flex';
            const onConfirm = () => { cleanup(); resolve(true); };
            const onCancel  = () => { cleanup(); resolve(false); };
            const onOverlay = (e) => { if (e.target === deleteModal) { cleanup(); resolve(false); } };
            function cleanup() {
                deleteModal.style.display = 'none';
                document.getElementById('modal-confirm').removeEventListener('click', onConfirm);
                document.getElementById('modal-cancel').removeEventListener('click', onCancel);
                deleteModal.removeEventListener('click', onOverlay);
            }
            document.getElementById('modal-confirm').addEventListener('click', onConfirm);
            document.getElementById('modal-cancel').addEventListener('click', onCancel);
            deleteModal.addEventListener('click', onOverlay);
        });
    }

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
                        if (data.node === 'coordinator' && data.output.frustration_level !== undefined) {
                            updateFrustrationUI(data.output.frustration_level);
                        }
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

            if (data.frustration_level !== undefined) {
                updateFrustrationUI(data.frustration_level);
            }

            // Remove Loading & Add AI Response (with retrieved docs inline)
            removeLoadingIndicator(loadingId);
            addMessage(data.answer, 'ai', data.retrieved_docs);

            // Update Plan
            if (data.plan) {
                currentPlan.innerHTML = `<div class="plan-content">${marked.parse(data.plan)}</div>`;
                currentPlan.classList.add('updated');
                setTimeout(() => currentPlan.classList.remove('updated'), 1000);
            }

            // Update Session Stats Docs Count
            if (data.retrieved_docs && data.retrieved_docs.length > 0) {
                docCount = data.retrieved_docs.length;
                statDocs.innerText = docCount;
            }

            // 세션 목록 및 약점 목록 갱신
            await loadSessions();
            if (data.tool_results && data.tool_results.length > 0) {
                if (data.tool_results.some(r => r.tool === 'schedule_review' && r.ok)) await fetchSchedules();
            }

            // 비동기 백그라운드 진단 결과 반영을 위해 2초 후 추가 갱신
            setTimeout(async () => {
                await fetchSchedules();
            }, 2000);


        } catch (error) {
            console.error(error);
            removeLoadingIndicator(loadingId);
            addMessage('죄송합니다. 서버와 통신 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.', 'ai');
        } finally {
            isThinking = false;
        }
    }

    function addMessage(text, role, retrievedDocs = []) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'msg-avatar';
        avatarDiv.innerText = role === 'ai' ? 'S' : (role === 'system' ? '⚙️' : 'U');

        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role === 'system') {
            contentDiv.innerHTML = `<span class="system-text">${marked.parse(text)}</span>`;
        } else {
            contentDiv.innerHTML = role === 'ai' ? marked.parse(text) : text.replace(/\n/g, '<br>');
        }
        
        bodyDiv.appendChild(contentDiv);

        // Add RAG references if it is AI and retrievedDocs exist
        if (role === 'ai' && retrievedDocs && retrievedDocs.length > 0) {
            const refsDiv = document.createElement('div');
            refsDiv.className = 'message-references';

            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'references-toggle';
            toggleBtn.innerHTML = `
                <svg class="ref-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/></svg>
                <span>참조한 강의 자료 (${retrievedDocs.length}개)</span>
                <span class="ref-arrow">▼</span>
            `;

            const listDiv = document.createElement('div');
            listDiv.className = 'references-list';
            listDiv.style.display = 'none';

            listDiv.innerHTML = retrievedDocs.map((doc, idx) => {
                const sourceName = doc.metadata?.source || '강의 자료 일부';
                const pageNum = doc.metadata?.page || 1;
                const docText = doc.text || '';
                return `
                    <div class="ref-item" data-index="${idx}">
                        <div class="ref-item-header">
                            <span class="ref-item-title" title="${sourceName}">${sourceName}</span>
                            <span class="ref-item-page">p.${pageNum}</span>
                            <span class="ref-expand-icon">▼</span>
                        </div>
                        <div class="ref-item-content">${docText}</div>
                    </div>
                `;
            }).join('');

            // Toggle whole list
            toggleBtn.addEventListener('click', () => {
                const isExpanded = listDiv.style.display === 'flex';
                listDiv.style.display = isExpanded ? 'none' : 'flex';
                toggleBtn.classList.toggle('active', !isExpanded);
                scrollToBottom();
            });

            // Toggle individual item content
            listDiv.querySelectorAll('.ref-item-header').forEach(header => {
                header.addEventListener('click', () => {
                    const item = header.parentElement;
                    item.classList.toggle('active');
                });
            });

            refsDiv.appendChild(toggleBtn);
            refsDiv.appendChild(listDiv);
            bodyDiv.appendChild(refsDiv);
        }

        msgDiv.appendChild(avatarDiv);
        msgDiv.appendChild(bodyDiv);
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
        'coordinator': '🔍 질문 분석 및 문맥 이해 & 유형 분류',
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
        if (node === 'coordinator') {
            const routeText = output.next_step === 'planner' ? '개념 학습 (PLAN)' : '일반 대화 (DIRECT)';
            detail = `재구성 결과: "${output.contextualized_query}"<br>판별 결과: ${routeText}`;
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

    function updateFrustrationUI(level) {
        frustrationLevel = level;
        let displayVal = frustrationLevel;
        let levelClass = 'level-safe';
        if (frustrationLevel >= 3) {
            displayVal = `${frustrationLevel} 😰`;
            levelClass = 'level-danger';
        } else if (frustrationLevel >= 1) {
            displayVal = `${frustrationLevel} ⚡`;
            levelClass = 'level-warning';
        }
        if (statFrustration) {
            statFrustration.innerText = displayVal;
            statFrustration.className = `stat-value ${levelClass}`;
        }
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
        chatMessages.innerHTML = '';
        welcomeBanner.style.display = 'flex';
        messages = [];
        sessionId = null;
        turnCount = 0;
        docCount = 0;
        statTurns.innerText = '0';
        statDocs.innerText = '0';
        updateFrustrationUI(0);
        currentPlan.innerHTML = '<div class="plan-empty"><p>질문하면 AI가<br>학습 계획을 세웁니다</p></div>';
        loadSessions();
        fetchSchedules();
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



    // Fetch and render review schedules
    async function fetchSchedules() {
        try {
            const response = await fetch('/schedules');
            if (!response.ok) throw new Error('Failed to fetch schedules');
            const data = await response.json();
            const items = data.schedules || [];

            if (items.length === 0) {
                scheduleList.innerHTML = '<div class="weakness-empty">등록된 복습 일정 없음</div>';
                return;
            }

            scheduleList.innerHTML = items.map(s => {
                const dt = new Date(s.review_at);
                const dateStr = dt.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
                const desc = s.description || '복습 일정';
                return `
                    <div class="weakness-item" data-id="${s.id}">
                        <div class="weakness-item-body">
                            <span class="weakness-concept" title="${desc}">${desc}</span>
                            <span class="weakness-sev sev-low">${dateStr}</span>
                        </div>
                        <button class="weakness-delete-btn" data-id="${s.id}" title="삭제">
                            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>
                        </button>
                    </div>
                `;
            }).join('');

            scheduleList.querySelectorAll('.weakness-delete-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const id = parseInt(btn.dataset.id);
                    await fetch(`/schedules/${id}`, { method: 'DELETE' });
                    await fetchSchedules();
                });
            });
        } catch (error) {
            console.error('Error fetching schedules:', error);
        }
    }

    // Initial fetch of registered documents
    fetchRegisteredDocuments();
    fetchSchedules();

    // 세션 목록 불러오기
    async function loadSessions() {
        try {
            const response = await fetch('/sessions');
            if (!response.ok) return;
            const data = await response.json();
            const sessions = data.sessions || [];

            if (sessions.length === 0) {
                sessionList.innerHTML = '<div class="session-empty">대화 기록 없음</div>';
                return;
            }

            sessionList.innerHTML = sessions.map(s => {
                const date = new Date(s.updated_at).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
                const title = s.title || '새 대화';
                const isActive = s.id === sessionId ? 'active' : '';
                return `
                    <div class="session-item ${isActive}" data-id="${s.id}">
                        <div class="session-item-body">
                            <div class="session-item-title">${title}</div>
                            <div class="session-item-date">${date}</div>
                        </div>
                        <button class="session-delete-btn" data-id="${s.id}" title="삭제">
                            <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>
                        </button>
                    </div>
                `;
            }).join('');

            sessionList.querySelectorAll('.session-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.session-delete-btn')) return;
                    loadSession(item.dataset.id);
                });
            });

            sessionList.querySelectorAll('.session-delete-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const id = btn.dataset.id;
                    if (!await showDeleteModal()) return;
                    await fetch(`/sessions/${id}`, { method: 'DELETE' });
                    if (sessionId === id) {
                        chatMessages.innerHTML = '';
                        if (welcomeBanner) welcomeBanner.style.display = 'flex';
                        messages = [];
                        sessionId = null;
                        turnCount = 0;
                        statTurns.innerText = '0';
                    }
                    await loadSessions();
                });
            });
        } catch (e) {
            console.error('Failed to load sessions:', e);
        }
    }

    // 특정 세션 대화 내역 복원
    async function loadSession(id) {
        try {
            const response = await fetch(`/sessions/${id}/messages`);
            if (!response.ok) return;
            const data = await response.json();
            const msgs = data.messages || [];

            // 화면 초기화
            chatMessages.innerHTML = '';
            if (welcomeBanner) welcomeBanner.style.display = 'none';
            messages = [];
            turnCount = 0;
            sessionId = id;

            msgs.forEach(m => {
                if (m.role === 'user' || m.role === 'assistant') {
                    addMessage(m.content, m.role === 'assistant' ? 'ai' : 'user');
                    messages.push({ role: m.role, content: m.content });
                    if (m.role === 'user') turnCount++;
                }
            });

            statTurns.innerText = turnCount;
            await loadSessions();
            await fetchSchedules();
        } catch (e) {
            console.error('Failed to load session:', e);
        }
    }

    // 페이지 로드 시 세션 목록 표시
    loadSessions();
});

