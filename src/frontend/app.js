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
    const quickStartChips = document.getElementById('quick-start-chips');
    const quickStartEmpty = document.getElementById('quick-start-empty');
    const attachBtn = document.getElementById('attach-btn');
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadStatus = document.getElementById('upload-status');
    const uploadStatusTitle = document.getElementById('upload-status-title');
    const uploadStatusMeta = document.getElementById('upload-status-meta');
    const registeredDocsList = document.getElementById('registered-docs-list');
    const sessionList = document.getElementById('session-list');
    const scheduleList = document.getElementById('schedule-list');
    const weaknessesList = document.getElementById('weaknesses-list');
    const openReportBtn = document.getElementById('open-report-btn');
    const reportModal = document.getElementById('report-modal');
    const closeReportBtn = document.getElementById('close-report-btn');
    const reportModalBody = document.getElementById('report-modal-body');
    const generateReportBtn = document.getElementById('generate-report-btn');

    // State
    let messages = [];
    let socraticDepth = 1;
    let sessionId = null;
    let isThinking = false;
    let turnCount = 0;
    let docCount = 0;
    let frustrationLevel = 0;
    let isPdfUploading = false;
    let uploadTimer = null;
    let uploadStartedAt = 0;

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
        const isCollapsed = sidebar.classList.contains('collapsed');
        sidebarOpenBtn.style.display = isCollapsed ? 'flex' : 'none';
        document.body.classList.toggle('sidebar-collapsed', isCollapsed);
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
    const depthAvatars = { 0: '💡', 1: '🔍', 2: '🔥' };
    const depthClasses = { 0: 'depth-light', 1: 'depth-standard', 2: 'depth-deep' };
    depthBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            socraticDepth = parseInt(btn.dataset.depth);
            updateDepthUI(socraticDepth);
        });
    });

    function updateDepthUI(depth) {
        depthBtns.forEach(b => {
            b.classList.remove('active');
            if (parseInt(b.dataset.depth) === depth) {
                b.classList.add('active');
            }
        });
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    async function loadPersonalizedChips() {
        if (!quickStartChips || !quickStartEmpty) return;
        try {
            const response = await fetch('/recommend-chips');
            if (!response.ok) throw new Error('Failed to load chips');
            const data = await response.json();
            const chips = Array.isArray(data.chips) ? data.chips : [];

            if (chips.length === 0) {
                quickStartChips.innerHTML = '';
                quickStartChips.style.display = 'none';
                quickStartEmpty.style.display = 'block';
                return;
            }

            quickStartEmpty.style.display = 'none';
            quickStartChips.style.display = 'flex';
            quickStartChips.innerHTML = chips.map(chip => {
                const typeClass = chip.type === 'schedule' ? 'chip-schedule' : 'chip-weakness';
                const label = escapeHtml(chip.label || '추천 학습');
                const message = escapeHtml(chip.message || '복습 도와줘');
                const due = chip.due ? `<span class="chip-badge">${escapeHtml(chip.due)}</span>` : '';
                return `
                    <button class="chip ${typeClass}" data-msg="${message}">
                        <span class="chip-text">${label}</span>
                        ${due}
                    </button>
                `;
            }).join('');
        } catch (error) {
            console.error('Error loading personalized chips:', error);
            quickStartChips.innerHTML = '';
            quickStartChips.style.display = 'none';
            quickStartEmpty.style.display = 'block';
        }
    }

    if (quickStartChips) {
        quickStartChips.addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (!chip) return;
            userInput.value = chip.dataset.msg || '';
            userInput.dispatchEvent(new Event('input'));
            sendMessage();
        });
    }

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

        // 선택된 문서 수집
        const checkedBoxes = Array.from(document.querySelectorAll('.doc-checkbox:checked'));
        const selectedDocs = checkedBoxes.map(cb => cb.value);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [...messages, { role: 'user', content: text }],
                    socratic_depth: socraticDepth,
                    session_id: sessionId,
                    selected_docs: selectedDocs
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
                        if (data.node === 'router' && data.output.frustration_level !== undefined) {
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

            if (data.socratic_depth !== undefined && data.socratic_depth !== socraticDepth) {
                socraticDepth = data.socratic_depth;
                updateDepthUI(socraticDepth);
                addMessage(`🤖 대화 맥락을 반영하여 학습 깊이를 자동으로 조절했습니다.`, 'system');
            }

            // Remove Loading & Add AI Response (with retrieved docs inline)
            removeLoadingIndicator(loadingId);
            addMessage(data.answer, 'ai', data.retrieved_docs);

            // Render Quiz UI if quiz_data exists and has items
            if (data.quiz_data && Array.isArray(data.quiz_data) && data.quiz_data.length > 0) {
                renderQuizUI(data.quiz_data);
            }

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
                if (data.tool_results.some(r => r.tool === 'save_weakness' && r.ok)) await fetchWeaknesses();
            }

            // 비동기 백그라운드 진단 결과 반영을 위해 2초 후 추가 갱신
            setTimeout(async () => {
                await fetchSchedules();
                await fetchWeaknesses();
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
        if (role === 'ai') {
            avatarDiv.innerText = depthAvatars[socraticDepth] ?? '🔍';
            avatarDiv.classList.add(depthClasses[socraticDepth] ?? 'depth-standard');
        } else {
            avatarDiv.innerText = role === 'system' ? '⚙️' : 'U';
        }

        const bodyDiv = document.createElement('div');
        bodyDiv.className = 'message-body';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role === 'system') {
            contentDiv.innerHTML = `<span class="system-text">${marked.parse(text)}</span>`;
        } else {
            contentDiv.innerHTML = role === 'ai' ? marked.parse(text.trim()) : text.replace(/\n/g, '<br>');
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
        'router':          '🔍 의도 분류 및 쿼리 재작성',
        'retrieval_agent': '📚 강의 자료 검색',
        'socratic_agent':  '🧠 소크라테스 응답 생성',
        'tool_agent':      '🛠️ 도구 실행',
        'composer':        '✍️ 최종 응답 조합',
        'reviewer':        '⚖️ 품질 검증',
        'responder':       '💬 일반 대화 응답',
    };

    const routeLabels = {
        'learn':  '학습 (learn)',
        'tools':  '도구 실행 (tools)',
        'escape': '정답 직접 제공 (escape)',
        'chat':   '일반 대화 (chat)',
    };

    function updateProgressStep(loadingId, node, output) {
        const stepsContainer = document.getElementById(`${loadingId}-steps`);
        if (!stepsContainer) return;

        let detail = '';
        if (node === 'router') {
            const route = output.route || '';
            const query = output.rewritten_query || '';
            const routeText = routeLabels[route] || route;
            detail = `분류: <strong>${routeText}</strong><br>재작성: "${escapeHtml(query)}"`;
        } else if (node === 'retrieval_agent') {
            const count = output.retrieved_docs ? output.retrieved_docs.length : 0;
            detail = `${count}개 자료 참조`;
        } else if (node === 'socratic_agent') {
            const tr = output.tutor_response || '';
            // Strip XML tags for preview
            const cleanText = tr.replace(/<[^>]*>?/gm, '').trim();
            detail = cleanText.length > 60 ? `초안: "${escapeHtml(cleanText.substring(0, 60))}..."` : `초안: "${escapeHtml(cleanText)}"`;
        } else if (node === 'tool_agent') {
            const result = output.tool_result;
            if (result && result.tool_results && result.tool_results.length > 0) {
                const tools = result.tool_results.map(t => t.tool).join(", ");
                detail = `사용된 도구: <strong>${tools}</strong>`;
            } else {
                detail = `사용된 도구 없음`;
            }
        } else if (node === 'composer') {
            const resp = output.response || '';
            const cleanResp = resp.replace(/<[^>]*>?/gm, '').trim();
            detail = cleanResp.length > 60 ? `최종: "${escapeHtml(cleanResp.substring(0, 60))}..."` : `최종: "${escapeHtml(cleanResp)}"`;
        } else if (node === 'reviewer') {
            const passed = output.evaluation?.pass;
            const retry = output.retry_count || 0;
            const fb = output.evaluation?.feedback || '';
            if (passed === false) {
                detail = `검증 실패 (재시도 ${retry}회)<br><span style="color: #ef4444; font-size: 0.7rem;">의견: "${escapeHtml(fb)}"</span>`;
            } else {
                detail = `검증 통과`;
            }
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
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            sendMessage();
        }
    });

    function clearUploadTimer() {
        if (uploadTimer) {
            clearInterval(uploadTimer);
            uploadTimer = null;
        }
    }

    function setUploadStatus(state, title, meta) {
        if (!uploadStatus || !uploadStatusTitle || !uploadStatusMeta) return;
        uploadStatus.style.display = 'block';
        uploadStatus.classList.remove('done', 'error');
        if (state === 'done') uploadStatus.classList.add('done');
        if (state === 'error') uploadStatus.classList.add('error');
        uploadStatusTitle.textContent = title;
        uploadStatusMeta.textContent = meta;
    }

    function hideUploadStatus(delayMs = 0) {
        if (!uploadStatus) return;
        window.setTimeout(() => {
            uploadStatus.style.display = 'none';
            uploadStatus.classList.remove('done', 'error');
        }, delayMs);
    }

    function startUploadStatus(fileName) {
        uploadStartedAt = Date.now();
        const escapedName = fileName || 'PDF';
        setUploadStatus('loading', `${escapedName} 처리 중...`, '업로드 시작됨 · 보통 2~3분 소요됩니다.');
        clearUploadTimer();
        uploadTimer = setInterval(() => {
            const elapsedSec = Math.floor((Date.now() - uploadStartedAt) / 1000);
            const mm = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
            const ss = String(elapsedSec % 60).padStart(2, '0');
            setUploadStatus('loading', `${escapedName} 처리 중...`, `진행 시간 ${mm}:${ss} · 텍스트 추출/임베딩 중일 수 있습니다.`);
        }, 1000);
    }

    function finishUploadStatus(isSuccess, message) {
        clearUploadTimer();
        if (isSuccess) {
            setUploadStatus('done', '처리 완료', message);
            hideUploadStatus(2500);
        } else {
            setUploadStatus('error', '처리 실패', message);
        }
    }
    
    // PDF Upload Handling
    if (attachBtn && pdfUpload) {
        attachBtn.addEventListener('click', () => {
            if (isPdfUploading) return;
            pdfUpload.click();
        });
        
        pdfUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file || isPdfUploading) return;
            isPdfUploading = true;
            attachBtn.disabled = true;
            startUploadStatus(file.name);
            
            // Show system message for upload starting
            addMessage(`📄 **${file.filename || file.name}** 파일 업로드를 시작합니다. 처리까지 2~3분 정도 걸릴 수 있어요.`, 'system');
            
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
                    finishUploadStatus(true, `${data.filename} 은 이미 등록되어 있어 중복 처리를 건너뛰었습니다.`);
                } else {
                    addMessage(`✅ **${data.filename}** 등록 완료! (${data.chunks_added}개의 지식 조각 추출)`, 'system');
                    finishUploadStatus(true, `${data.filename} 처리 완료 · ${data.chunks_added}개 지식 조각이 반영되었습니다.`);
                }
                
                // Update registered documents list
                await fetchRegisteredDocuments();

            } catch (error) {
                console.error(error);
                addMessage(`❌ 업로드 실패: ${error.message}`, 'system');
                finishUploadStatus(false, `오류: ${error.message}`);
            } finally {
                // Clear input so same file can be uploaded again if needed
                pdfUpload.value = '';
                attachBtn.disabled = false;
                isPdfUploading = false;
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
        fetchWeaknesses();
        loadPersonalizedChips();
    });

    // Fetch and render registered PDF documents
    async function fetchRegisteredDocuments() {
        try {
            const response = await fetch(`/documents?ts=${Date.now()}`, {
                cache: 'no-store'
            });
            if (!response.ok) throw new Error('Failed to fetch documents');
            const data = await response.json();
            
            if (data.documents && data.documents.length > 0) {
                registeredDocsList.innerHTML = data.documents.map(filename => `
                    <div class="doc-item" data-filename="${filename}">
                        <div class="doc-item-left">
                            <input type="checkbox" class="doc-checkbox" value="${filename}" checked title="검색 대상 포함">
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
        } finally {
            loadPersonalizedChips();
        }
    }

    // Fetch and render weaknesses
    async function fetchWeaknesses() {
        try {
            const response = await fetch('/weaknesses');
            if (!response.ok) throw new Error('Failed to fetch weaknesses');
            const data = await response.json();
            const items = data.weaknesses || [];

            if (items.length === 0) {
                weaknessesList.innerHTML = '<div class="weakness-empty">등록된 약점 없음</div>';
                return;
            }

            weaknessesList.innerHTML = items.map(w => {
                const sevClass = w.severity <= 2 ? 'sev-low' : w.severity <= 3 ? 'sev-mid' : 'sev-high';
                const sevLabel = ['', '낮음', '낮음', '중간', '높음', '매우높음'][w.severity] || '중간';
                const createdDate = new Date(w.created_at).toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
                return `
                    <div class="weakness-item" data-id="${w.id}">
                        <div class="weakness-item-body">
                            <span class="weakness-concept" title="${w.concept}">${w.concept}</span>
                            <span class="weakness-sev ${sevClass}">${sevLabel}</span>
                        </div>
                        <button class="weakness-delete-btn" data-id="${w.id}" title="삭제">
                            <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>
                        </button>
                    </div>
                `;
            }).join('');

            weaknessesList.querySelectorAll('.weakness-delete-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const id = parseInt(btn.dataset.id);
                    await fetch(`/weaknesses/${id}`, { method: 'DELETE' });
                    await fetchWeaknesses();
                    await fetchSchedules();
                });
            });

            // Add click to view details
            weaknessesList.querySelectorAll('.weakness-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (e.target.closest('.weakness-delete-btn')) return;
                    const weaknessId = item.dataset.id;
                    const weakness = items.find(w => w.id == weaknessId);
                    if (weakness) {
                        addMessage(`📌 **${weakness.concept}**\n\n심각도: ${['', '⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'][weakness.severity]}\n\n상세: ${weakness.details || '별도 기록 없음'}\n\n등록일: ${new Date(weakness.created_at).toLocaleDateString('ko-KR')}`, 'system');
                    }
                });
            });
        } catch (error) {
            console.error('Error fetching weaknesses:', error);
        } finally {
            loadPersonalizedChips();
        }
    }

    function renderReportModal(data) {
        const weaknesses = data.weaknesses || [];
        const strengths = data.strengths || [];
        const prioritizedWeaknesses = data.prioritized_weaknesses || [];
        const profile = data.profile_summary || {};
        const stats = data.stats || {};

        const weaknessById = new Map(weaknesses.map(w => [Number(w.id), w]));
        const topPriority = prioritizedWeaknesses.length > 0
            ? prioritizedWeaknesses.slice(0, 5).map(p => ({
                ...p,
                ...weaknessById.get(Number(p.id)),
            }))
            : [...weaknesses]
                .sort((a, b) => Number(b.severity || 1) - Number(a.severity || 1))
                .slice(0, 5)
                .map(w => ({ ...w, priority_score: Number(w.severity || 1) }));

        const maxPriorityScore = Math.max(
            1,
            ...topPriority.map(w => Number(w.priority_score || w.severity || 1))
        );

        const strengthsHtml = strengths.length > 0
            ? strengths.slice(0, 8).map(s => `
                <div class="report-card-item strength">
                    <span class="report-item-title">${escapeHtml(s.concept)}</span>
                    <span class="report-item-sub">${escapeHtml(s.details || '이해가 잘 된 개념')}</span>
                </div>
            `).join('')
            : '<div class="report-empty">아직 기록된 강점이 없습니다.</div>';

        const weaknessesHtml = topPriority.length > 0
            ? topPriority.map(w => {
                const severity = Math.min(5, Math.max(1, Number(w.severity || 1)));
                const rawScore = Number(w.priority_score || w.severity || 1);
                const score = Number.isFinite(rawScore) ? rawScore : Number(w.severity || 1);
                const ratio = Math.max(12, Math.round((score / maxPriorityScore) * 100));
                return `
                <div class="report-card-item weakness">
                    <div class="report-item-main">
                        <span class="report-item-title">${escapeHtml(w.concept)}</span>
                        <span class="report-severity sev-${severity}">심각도 ${escapeHtml(w.severity || 1)}</span>
                    </div>
                    <div class="report-priority-row">
                        <span class="report-priority-score">${escapeHtml(score.toFixed(2))}</span>
                        <div class="report-priority-track">
                            <div class="report-priority-fill" style="width:${ratio}%"></div>
                        </div>
                    </div>
                    <div class="report-item-actions">
                        <button class="report-review-btn" data-concept="${escapeHtml(w.concept)}">복습하기</button>
                    </div>
                </div>
                `;
            }).join('')
            : '<div class="report-empty">현재 등록된 약점이 없습니다.</div>';

        const topCategories = (stats.top_categories || []).length > 0
            ? stats.top_categories.map(c => `<span class="report-chip">${escapeHtml(c)}</span>`).join(' ')
            : '<span class="report-empty-inline">분류 데이터 없음</span>';

        const trend = String(stats.weekly_trend || 'steady');
        const trendLabel = trend === 'worse'
            ? '악화'
            : trend === 'improving'
                ? '개선'
                : '유지';
        const trendClass = trend === 'worse'
            ? 'trend-worse'
            : trend === 'improving'
                ? 'trend-improving'
                : 'trend-steady';

        const recentWeak = Number(stats.recent_7d_weaknesses || 0);
        const prevWeak = Number(stats.previous_7d_weaknesses || 0);
        const diff = recentWeak - prevWeak;
        const diffText = diff > 0 ? `+${diff}` : `${diff}`;

        const actionTargets = topPriority.slice(0, 3);
        const actionHtml = actionTargets.length > 0
            ? actionTargets.map((w, idx) => `
                <button class="report-action-chip" data-concept="${escapeHtml(w.concept)}">
                    ${idx + 1}. ${escapeHtml(w.concept)} 복습 시작
                </button>
            `).join('')
            : '<div class="report-empty">실행할 약점 항목이 아직 없습니다.</div>';

        reportModalBody.innerHTML = `
            <section class="report-section">
                <h3>✅ 잘 이해하고 있는 것</h3>
                <div class="report-summary">${escapeHtml(profile.strengths_summary || '강점 요약이 아직 없습니다.')}</div>
                <div class="report-card-list">${strengthsHtml}</div>
            </section>

            <section class="report-section">
                <h3>🔴 보완이 필요한 것 (우선순위)</h3>
                <div class="report-summary">${escapeHtml(profile.weaknesses_summary || '약점 요약이 아직 없습니다.')}</div>
                <div class="report-card-list">${weaknessesHtml}</div>
            </section>

            <section class="report-section">
                <h3>📈 학습 통계</h3>
                <div class="report-stats-grid">
                    <div class="report-stat"><span class="k">총 약점</span><span class="v">${escapeHtml(stats.total_weaknesses || 0)}</span></div>
                    <div class="report-stat"><span class="k">총 강점</span><span class="v">${escapeHtml(stats.total_strengths || 0)}</span></div>
                    <div class="report-stat"><span class="k">최근 7일 약점</span><span class="v">${escapeHtml(stats.recent_7d_weaknesses || 0)}</span></div>
                </div>
                <div class="report-categories">${topCategories}</div>
                <div class="report-trend ${trendClass}">
                    <div class="report-trend-main">
                        <span class="report-trend-label">2주 추세</span>
                        <span class="report-trend-value">${trendLabel}</span>
                    </div>
                    <div class="report-trend-sub">최근 7일 ${recentWeak}건 / 이전 7일 ${prevWeak}건 (${diffText})</div>
                </div>
            </section>

            <section class="report-section">
                <h3>🎯 지금 바로 실행할 복습</h3>
                <div class="report-action-row">${actionHtml}</div>
            </section>

            <section class="report-section" id="report-generated-area"></section>
        `;

        reportModalBody.querySelectorAll('.report-review-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const concept = btn.dataset.concept;
                reportModal.style.display = 'none';
                userInput.value = `${concept} 개념을 소크라테스식으로 복습하고 싶어`;
                userInput.dispatchEvent(new Event('input'));
                sendMessage();
            });
        });

        reportModalBody.querySelectorAll('.report-action-chip').forEach(btn => {
            btn.addEventListener('click', () => {
                const concept = btn.dataset.concept;
                reportModal.style.display = 'none';
                userInput.value = `${concept} 개념을 핵심부터 점검하고 복습 퀴즈 3문항 내줘`;
                userInput.dispatchEvent(new Event('input'));
                sendMessage();
            });
        });
    }

    function normalizeReportMarkdown(raw) {
        const text = String(raw || '').replace(/\r\n/g, '\n').trim();
        return text
            .replace(/^(##\s+)/gm, '\n$1')
            .replace(/\n{3,}/g, '\n\n')
            .trim();
    }

    async function openReportModal() {
        if (!reportModal || !reportModalBody) return;
        reportModal.style.display = 'flex';
        reportModalBody.innerHTML = '<div class="report-loading">리포트 데이터를 불러오는 중...</div>';
        try {
            const response = await fetch('/report-data');
            if (!response.ok) throw new Error('Failed to fetch report data');
            const data = await response.json();
            renderReportModal(data);
        } catch (error) {
            console.error('Error loading report modal:', error);
            reportModalBody.innerHTML = '<div class="report-empty">리포트 데이터를 불러오지 못했습니다.</div>';
        }
    }

    async function generateMetacognitiveReport() {
        if (!generateReportBtn || !reportModalBody) return;
        const oldText = generateReportBtn.textContent;
        generateReportBtn.disabled = true;
        generateReportBtn.textContent = '생성 중...';

        try {
            const response = await fetch('/generate-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: 'default' })
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Report generation failed');
            }
            const data = await response.json();
            const generatedArea = document.getElementById('report-generated-area');
            if (generatedArea) {
                const normalizedBody = normalizeReportMarkdown(data.body || '');
                generatedArea.innerHTML = `
                    <h3>🧠 AI 생성 리포트</h3>
                    <div class="report-markdown">${marked.parse(normalizedBody)}</div>
                `;
            }
        } catch (error) {
            console.error('Error generating report:', error);
            const generatedArea = document.getElementById('report-generated-area');
            if (generatedArea) {
                generatedArea.innerHTML = `<div class="report-empty">리포트 생성 실패: ${escapeHtml(error.message)}</div>`;
            }
        } finally {
            generateReportBtn.disabled = false;
            generateReportBtn.textContent = oldText;
        }
    }

    if (openReportBtn) {
        openReportBtn.addEventListener('click', openReportModal);
    }
    if (closeReportBtn && reportModal) {
        closeReportBtn.addEventListener('click', () => {
            reportModal.style.display = 'none';
        });
        reportModal.addEventListener('click', (e) => {
            if (e.target === reportModal) {
                reportModal.style.display = 'none';
            }
        });
    }
    if (generateReportBtn) {
        generateReportBtn.addEventListener('click', generateMetacognitiveReport);
    }

    // Initial fetch of registered documents
    fetchRegisteredDocuments();
    fetchSchedules();
    fetchWeaknesses();
    loadPersonalizedChips();

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

    // Quiz UI Rendering
    function renderQuizUI(quizItems) {
        const quizContainer = document.createElement('div');
        quizContainer.className = 'quiz-container';
        quizContainer.innerHTML = `
            <div class="quiz-header">
                <h3>📝 문제를 풀어보세요!</h3>
                <p class="quiz-count">총 ${quizItems.length}문제</p>
            </div>
            <div class="quiz-questions" id="quiz-questions"></div>
            <div class="quiz-actions">
                <button id="submit-quiz-btn" class="submit-btn">채점하기</button>
            </div>
        `;

        const questionsDiv = quizContainer.querySelector('#quiz-questions');

        // Render each question with radio buttons
        quizItems.forEach((item, idx) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'quiz-question';
            questionDiv.innerHTML = `
                <div class="question-text">
                    <span class="question-num">${idx + 1}.</span>
                    <span>${item.question}</span>
                </div>
                <div class="question-options">
                    ${(item.options || []).map((opt, optIdx) => {
                        const letter = String.fromCharCode(65 + optIdx); // A, B, C, D
                        const optionId = `q${idx}_${letter}`;
                        return `
                            <div class="option">
                                <input 
                                    type="radio" 
                                    id="${optionId}" 
                                    name="q${idx}" 
                                    value="${letter}"
                                    class="option-radio"
                                >
                                <label for="${optionId}">
                                    <span class="option-letter">${letter}</span>
                                    <span class="option-text">${opt}</span>
                                </label>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
            questionsDiv.appendChild(questionDiv);
        });

        // Submit button event listener
        const submitBtn = quizContainer.querySelector('#submit-quiz-btn');
        submitBtn.addEventListener('click', async () => {
            // Collect answers
            const userAnswers = {};
            let answered = 0;

            for (let i = 0; i < quizItems.length; i++) {
                const radioButtons = document.querySelectorAll(`input[name="q${i}"]`);
                const selected = Array.from(radioButtons).find(r => r.checked);
                if (selected) {
                    userAnswers[i] = selected.value;
                    answered++;
                } else {
                    addMessage(`⚠️ ${i + 1}번 문제를 선택하지 않았습니다.`, 'system');
                    return;
                }
            }

            if (answered !== quizItems.length) {
                addMessage(`⚠️ 모든 문제를 선택해주세요.`, 'system');
                return;
            }

            // Disable button and show loading
            submitBtn.disabled = true;
            submitBtn.innerHTML = '채점 중...';

            try {
                const response = await fetch('/submit_quiz', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        quiz_items: quizItems,
                        user_answers: userAnswers
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Grading failed');
                }

                const result = await response.json();

                // Display grading result
                renderGradingResult(result, quizItems, userAnswers);

                // Re-enable button
                submitBtn.innerHTML = '다시 풀기';
                submitBtn.disabled = false;

            } catch (error) {
                console.error('Quiz submission error:', error);
                addMessage(`❌ 채점 중 오류가 발생했습니다: ${error.message}`, 'system');
                submitBtn.innerHTML = '채점하기';
                submitBtn.disabled = false;
            }
        });

        // Add quiz container to chat
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai';
        msgDiv.appendChild(quizContainer);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function renderGradingResult(result, quizItems, userAnswers) {
        const resultContainer = document.createElement('div');
        resultContainer.className = 'grading-result';

        const scorePercentage = result.score || 0;
        const statusEmoji = scorePercentage >= 80 ? '🎉' : scorePercentage >= 60 ? '👍' : '😞';

        resultContainer.innerHTML = `
            <div class="result-header ${scorePercentage >= 80 ? 'excellent' : scorePercentage >= 60 ? 'good' : 'poor'}">
                <span class="result-emoji">${statusEmoji}</span>
                <div class="result-score">
                    <span class="score-value">${result.score}점</span>
                    <span class="score-detail">${result.correct}/${result.total} 정답</span>
                </div>
            </div>
            <div class="result-message">${result.message}</div>
            ${result.suggest_weakness ? '<div class="weakness-suggestion">💡 이 주제를 보충학습으로 등록하시겠어요?</div>' : ''}
            <div class="result-details">
                ${result.details.map((detail, idx) => `
                    <div class="detail-item">
                        <span>${detail}</span>
                    </div>
                `).join('')}
            </div>
        `;

        // Add weakness suggestion button if needed
        if (result.suggest_weakness) {
            const weaknessBtn = document.createElement('button');
            weaknessBtn.className = 'weakness-suggestion-btn';
            weaknessBtn.innerHTML = '약점 등록하기';
            weaknessBtn.addEventListener('click', () => {
                userInput.value = '이 주제를 약점으로 등록해줄래?';
                userInput.dispatchEvent(new Event('input'));
                sendMessage();
            });
            resultContainer.appendChild(weaknessBtn);
        }

        const msgDiv = document.createElement('div');
        msgDiv.className = 'message ai';
        msgDiv.appendChild(resultContainer);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});


