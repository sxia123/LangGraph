document.addEventListener('DOMContentLoaded', () => {
  const statusPill = document.getElementById('statusPill');
  const statusText = document.getElementById('statusText');
  const tabs = document.querySelectorAll('.tab-btn');
  const messagesContainer = document.getElementById('messagesContainer');
  const chatForm = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const promptChips = document.querySelectorAll('.prompt-chip');

  let activePipeline = 'master';

  // 1. Check Server & LLM Connection Status
  async function checkStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.connection && data.connection.ok) {
        statusPill.querySelector('.status-dot').className = 'status-dot';
        statusText.textContent = `Active Model: ${data.model_name} (${data.provider.toUpperCase()})`;
      } else {
        statusPill.querySelector('.status-dot').className = 'status-dot error';
        statusText.textContent = `Offline Simulation (Endpoint: ${data.base_url})`;
      }
    } catch (err) {
      statusPill.querySelector('.status-dot').className = 'status-dot error';
      statusText.textContent = 'Backend Server Disconnected';
    }
  }

  checkStatus();
  setInterval(checkStatus, 30000);

  // 2. Tab Selection Handler
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activePipeline = tab.getAttribute('data-pipeline');
    });
  });

  // 3. Quick Prompt Chip Handlers
  promptChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-prompt');
      userInput.value = text;
      userInput.focus();
      autoResizeTextarea();
    });
  });

  // 4. Auto-resize Textarea
  function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
  }

  userInput.addEventListener('input', autoResizeTextarea);

  userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // 5. Submit Form Handler
  chatForm.addEventListener('submit', async e => {
    e.preventDefault();
    const promptText = userInput.value.trim();
    if (!promptText) return;

    // Append User Message
    appendUserMessage(promptText);

    userInput.value = '';
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // Show Typing / Executing Indicator
    const typingIndicatorId = appendTypingIndicator();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptText,
          pipeline: activePipeline
        })
      });

      removeMessage(typingIndicatorId);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      appendAgentResponse(data);
    } catch (err) {
      removeMessage(typingIndicatorId);
      appendErrorMessage(err.message);
    } finally {
      sendBtn.disabled = false;
      scrollToBottom();
    }
  });

  function appendUserMessage(text) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="message-sender">You • ${timeStr}</div>
      <div class="bubble">${escapeHtml(text)}</div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const id = 'typing_' + Date.now();
    const row = document.createElement('div');
    row.className = 'message-row agent';
    row.id = id;
    row.innerHTML = `
      <div class="message-sender">LangGraph Runtime • Executing Pipeline...</div>
      <div class="bubble typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
    return id;
  }

  function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendAgentResponse(data) {
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const steps = data.steps || [];
    const finalResp = data.final_response || 'Execution finished.';

    let thoughtsHtml = '';
    const allThoughts = [];
    steps.forEach(s => {
      if (s.thoughts && s.thoughts.length > 0) {
        s.thoughts.forEach(t => allThoughts.push(t));
      }
    });

    if (allThoughts.length > 0) {
      const thoughtItems = allThoughts.map(t => `
        <div class="thought-item">
          <span class="thought-agent">[${escapeHtml(t.agent || 'Agent')}]:</span>
          <span>${escapeHtml(t.thought || '')}</span>
        </div>
      `).join('');

      thoughtsHtml = `
        <div class="thoughts-drawer">
          <details>
            <summary class="thoughts-header">
              <span>🧠 Agent Thought Log (${allThoughts.length} steps)</span>
            </summary>
            <div class="thoughts-body">
              ${thoughtItems}
            </div>
          </details>
        </div>
      `;
    }

    const row = document.createElement('div');
    row.className = 'message-row agent';

    const parsedContent = typeof marked !== 'undefined' ? marked.parse(finalResp) : escapeHtml(finalResp);

    row.innerHTML = `
      <div class="message-sender">LangGraph Master Pipeline • ${timeStr}</div>
      <div class="bubble">
        ${parsedContent}
        ${thoughtsHtml}
      </div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function appendErrorMessage(msgText) {
    const row = document.createElement('div');
    row.className = 'message-row agent';
    row.innerHTML = `
      <div class="message-sender" style="color:#ef4444">System Warning</div>
      <div class="bubble" style="border-color: rgba(239,68,68,0.3)">
        <p>⚠️ <strong>Execution Note:</strong> ${escapeHtml(msgText)}</p>
      </div>
    `;
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function scrollToBottom() {
    messagesContainer.parentElement.scrollTop = messagesContainer.parentElement.scrollHeight;
  }

  function escapeHtml(str) {
    return str.replace(/[&<>"']/g, match => {
      const escape = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
      return escape[match];
    });
  }
});
