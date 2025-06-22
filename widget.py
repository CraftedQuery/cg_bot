"""
widget.py - Widget generation and serving
"""
import json
from fastapi.responses import PlainTextResponse

from .config import load_config


def generate_widget_js(tenant: str, agent: str) -> str:
    """Generate widget JavaScript with configuration"""
    
    cfg = load_config(tenant, agent)
    
    # Generate the JavaScript with proper escaping
    config_json = json.dumps(cfg)
    
    return f"""(function(){{
const p=new URLSearchParams(location.search);
const tenant=p.get('tenant')||'{tenant}';
const agent=p.get('agent')||'{agent}';
const sid=sessionStorage.getItem('cq_sid')||(()=>{{const r=Math.random().toString(36).slice(2);sessionStorage.setItem('cq_sid',r);return r}})();
const msgs=[];
function $(id){{return document.getElementById(id);}}

// Configuration from server
const config = {config_json};

// Create widget container
const container = document.createElement('div');
container.id = 'cq-widget-container';
document.body.appendChild(container);

// Features based on configuration
const features = {{
  typing: true,
  fileAttachments: config.enable_files || false,
  voiceInput: config.enable_voice || false,
  darkMode: config.enable_dark_mode || false,
  responseSpeech: config.enable_tts || false
}};

// Authentication mechanism
let authToken = localStorage.getItem('cq_auth_token');
const headers = {
  'Content-Type': 'application/json',
  'X-Session-Id': sid
};
if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

const circledNums = ['\u2460','\u2461','\u2462','\u2463','\u2464','\u2465','\u2466','\u2467','\u2468','\u2469'];

function initWidget() {{
  // Dynamic positioning based on config
  const position = config.widget_position || 'bottom-right';
  const [vPos, hPos] = position.split('-');
  
  // Size configuration
  const size = config.widget_size || 'medium';
  const sizes = {{
    small: {{ width: '300px', height: '400px' }},
    medium: {{ width: '350px', height: '500px' }},
    large: {{ width: '400px', height: '600px' }}
  }};
  
  const widgetSize = sizes[size] || sizes.medium;
  
  // Create chat interface with dynamic configuration
  container.innerHTML = `
    <div id="cq-chat-widget" style="--primary:${{config.primary_color}};--secondary:${{config.secondary_color}};width:${{widgetSize.width}};height:${{widgetSize.height}}">
      <div id="cq-header">
        <img src="${{config.avatar_url || '/default-avatar.png'}}" alt="${{config.bot_name}}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2232%22 height=%2232%22><circle cx=%2216%22 cy=%2216%22 r=%2215%22 fill=%22%23ddd%22/><text x=%2216%22 y=%2220%22 text-anchor=%22middle%22 font-size=%2216%22>🤖</text></svg>'">
        <h3>${{config.bot_name}}</h3>
        <div class="cq-controls">
          ${{features.darkMode ? '<button id="cq-dark-toggle" title="Toggle dark mode">🌓</button>' : ''}}
          <button id="cq-settings">⚙️</button>
          <button id="cq-minimize">—</button>
        </div>
      </div>
      <div id="cq-messages">
        ${{config.welcome_message ? `<div class="cq-welcome-message">${{config.welcome_message}}</div>` : ''}}
      </div>
      <div id="cq-typing" style="display:none">
        <div class="cq-dot"></div>
        <div class="cq-dot"></div>
        <div class="cq-dot"></div>
      </div>
      <div id="cq-footer">
        <div id="cq-attachments"></div>
        <textarea id="cq-input" placeholder="${{config.placeholder_text || 'Please ask your question...'}}"></textarea>
        <div class="cq-actions">
          ${{features.voiceInput ? '<button id="cq-mic" title="Voice input">🎤</button>' : ''}}
          ${{features.fileAttachments ? '<button id="cq-attach" title="Attach file">📎</button>' : ''}}
          <button id="cq-send">➤</button>
        </div>
        ${{features.fileAttachments ? '<input type="file" id="cq-file-input" style="display:none" multiple accept=".pdf,.txt,.md,.docx">' : ''}}
      </div>
      <div id="cq-settings-panel" style="display:none">
        <h4>Settings</h4>
        ${{features.darkMode ? '<label><input type="checkbox" id="cq-dark-mode"> Dark mode</label>' : ''}}
        ${{features.responseSpeech ? '<label><input type="checkbox" id="cq-speech"> Read responses aloud</label>' : ''}}
        <h4>Sources</h4>
        <div id="cq-sources"></div>
      </div>
    </div>
    <button id="cq-launcher" style="display:none">💬</button>
  `;
  
  // Set up event handlers
  setupEventHandlers();
  
  // Apply positioning
  container.style.position = 'fixed';
  container.style.zIndex = '10000';
  
  if (hPos === 'right') container.style.right = '20px';
  else container.style.left = '20px';
  
  if (vPos === 'bottom') container.style.bottom = '20px';
  else container.style.top = '20px';
  
  // Auto-open if configured
  if (config.auto_open) {{
    showWidget();
  }} else {{
    minimizeWidget();
  }}
}}

function setupEventHandlers() {{
  $('cq-send').addEventListener('click', sendMessage);
  $('cq-input').addEventListener('keypress', e => {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      sendMessage();
    }}
  }});
  
  $('cq-settings').addEventListener('click', toggleSettings);
  $('cq-minimize').addEventListener('click', minimizeWidget);
  $('cq-launcher').addEventListener('click', showWidget);
  
  // Feature-specific handlers
  if (features.voiceInput && $('cq-mic')) {{
    $('cq-mic').addEventListener('click', startVoiceInput);
  }}
  
  if (features.fileAttachments && $('cq-attach')) {{
    $('cq-attach').addEventListener('click', () => $('cq-file-input').click());
    $('cq-file-input').addEventListener('change', handleFileAttachment);
  }}
  
  if (features.darkMode) {{
    if ($('cq-dark-mode')) {{
      $('cq-dark-mode').addEventListener('change', e => setDarkMode(e.target.checked));
    }}
    if ($('cq-dark-toggle')) {{
      $('cq-dark-toggle').addEventListener('click', toggleDarkMode);
    }}
  }}
  
  if (features.responseSpeech && $('cq-speech')) {{
    $('cq-speech').addEventListener('change', e => features.responseSpeech = e.target.checked);
  }}
}}

function sendMessage() {{
  const input = $('cq-input');
  const text = input.value.trim();
  if (!text) return;
  
  addMessage('user', text);
  input.value = '';
  
  // Show typing indicator
  if (features.typing) $('cq-typing').style.display = 'flex';
  
  // Send to backend
  fetch(`/chat?tenant=${{tenant}}&agent=${{agent}}`, {{
    method: 'POST',
    headers: headers,
    body: JSON.stringify({{messages: msgs}})
  }})
  .then(r => r.json())
  .then(data => {
    if (features.typing) $('cq-typing').style.display = 'none';
    addMessage('assistant', data.reply, data.sources);
    updateSources(data.sources);
    
    // Text-to-speech if enabled
    if (features.responseSpeech) {{
      const speech = new SpeechSynthesisUtterance(data.reply);
      window.speechSynthesis.speak(speech);
    }}
  }})
  .catch(error => {{
    if (features.typing) $('cq-typing').style.display = 'none';
    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
    console.error('Chat error:', error);
  }});
}}

function addMessage(role, content, sources = []) {
  msgs.push({ role, content });

  const msgDiv = document.createElement('div');
  msgDiv.className = `cq-message cq-${role}`;

  let bubble = `<div class="cq-avatar">${role === 'user' ? '👤' : '🤖'}</div>`;
  bubble += `<div class="cq-bubble">${formatMessage(content)}`;
  if (role === 'assistant' && sources && sources.length) {
    sources.slice(0, 6).forEach((s, i) => {
      bubble += ` <a href="${s.source}" target="_blank" class="cq-cite-num">${circledNums[i]}</a>`;
    });
  }
  bubble += '</div>';
  msgDiv.innerHTML = bubble;

  if (role === 'assistant' && sources && sources.length) {
    const citeDiv = document.createElement('div');
    citeDiv.className = 'cq-cite-window';
    let idx = 0;
    citeDiv.innerHTML = `<a class="cq-citation-link" href="${sources[0].source}" target="_blank">${sources[0].source}</a>${sources.length > 1 ? '<span>→</span>' : ''}`;
    if (sources.length > 1) {
      citeDiv.querySelector('span').addEventListener('click', () => {
        idx = (idx + 1) % sources.length;
        const src = sources[idx];
        citeDiv.querySelector('a').href = src.source;
        citeDiv.querySelector('a').textContent = src.source;
      });
    }
    msgDiv.appendChild(citeDiv);
  }

  $('cq-messages').appendChild(msgDiv);
  $('cq-messages').scrollTop = $('cq-messages').scrollHeight;
}

function formatMessage(text) {{
  return text
    .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
    .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\\n/g, '<br>');
}}

function startVoiceInput() {{
  if (!('webkitSpeechRecognition' in window)) {{
    alert('Voice input not supported in your browser');
    return;
  }}
  
  const recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;
  
  recognition.onstart = () => {{
    $('cq-mic').style.background = '#f44336';
  }};
  
  recognition.onend = () => {{
    $('cq-mic').style.background = '';
  }};
  
  recognition.onresult = function(event) {{
    const transcript = event.results[0][0].transcript;
    $('cq-input').value = transcript;
  }};
  
  recognition.start();
}}

function handleFileAttachment(event) {{
  const files = Array.from(event.target.files);
  if (files.length === 0) return;
  
  const attachmentsDiv = $('cq-attachments');
  attachmentsDiv.innerHTML = '';
  
  files.forEach(file => {{
    const fileDiv = document.createElement('div');
    fileDiv.className = 'cq-attachment';
    fileDiv.innerHTML = `
      <span>📎 ${{file.name}}</span>
      <button onclick="this.parentElement.remove()">×</button>
    `;
    attachmentsDiv.appendChild(fileDiv);
  }});
  
  // Note: File upload implementation would go here
  console.log('Files selected:', files);
}}

function updateSources(sources) {{
  const sourcesDiv = $('cq-sources');
  sourcesDiv.innerHTML = '';
  
  if (sources && sources.length > 0) {{
    const sourceList = document.createElement('ul');
    sources.forEach(src => {{
      const li = document.createElement('li');
      li.innerHTML = `<a href="${{src.source}}" target="_blank">${{src.source}}</a>`;
      sourceList.appendChild(li);
    }});
    sourcesDiv.appendChild(sourceList);
  }}
}}

function toggleSettings() {{
  const panel = $('cq-settings-panel');
  panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}}

function minimizeWidget() {{
  $('cq-chat-widget').style.display = 'none';
  $('cq-launcher').style.display = 'block';
}}

function showWidget() {{
  $('cq-chat-widget').style.display = 'flex';
  $('cq-launcher').style.display = 'none';
}}

function setDarkMode(enabled) {{
  const widget = $('cq-chat-widget');
  if (enabled) {{
    widget.classList.add('cq-dark');
  }} else {{
    widget.classList.remove('cq-dark');
  }}
  localStorage.setItem('cq_dark_mode', enabled);
}}

function toggleDarkMode() {{
  const current = localStorage.getItem('cq_dark_mode') === 'true';
  setDarkMode(!current);
  if ($('cq-dark-mode')) {{
    $('cq-dark-mode').checked = !current;
  }}
}}

// CSS styles
const style = document.createElement('style');
style.textContent = `{generate_widget_css(cfg)}`;
document.head.appendChild(style);

// Initialize
if (document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', initWidget);
}} else {{
  initWidget();
}}

// Load saved preferences
const savedDarkMode = localStorage.getItem('cq_dark_mode') === 'true';
if (savedDarkMode && features.darkMode) {{
  setTimeout(() => setDarkMode(true), 100);
}}

}})();"""


def generate_widget_css(config):
    """Generate CSS for the widget"""
    primary_color = config.get('primary_color', '#1E88E5')
    secondary_color = config.get('secondary_color', '#FFFFFF')
    
    return f"""
  #cq-widget-container {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  
  #cq-launcher {{
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: {primary_color};
    color: white;
    border: none;
    font-size: 24px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
  }}
  
  #cq-launcher:hover {{
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
  }}
  
  #cq-chat-widget {{
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    background: white;
    transition: all 0.3s ease;
    border: 1px solid #e0e0e0;
  }}
  
  #cq-chat-widget.cq-dark {{
    background: #2d2d2d;
    border-color: #404040;
    color: white;
  }}
  
  #cq-header {{
    background: {primary_color};
    color: white;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  
  #cq-header img {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    object-fit: cover;
  }}
  
  #cq-header h3 {{
    margin: 0;
    flex: 1;
    font-size: 16px;
    font-weight: 600;
  }}
  
  .cq-controls {{
    display: flex;
    gap: 8px;
  }}
  
  .cq-controls button {{
    background: rgba(255,255,255,0.2);
    border: none;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
  }}
  
  .cq-controls button:hover {{
    background: rgba(255,255,255,0.3);
  }}
  
  #cq-messages {{
    flex: 1;
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  
  .cq-welcome-message {{
    text-align: center;
    padding: 12px;
    background: #f0f7ff;
    border-radius: 8px;
    color: #1976d2;
    font-style: italic;
    margin-bottom: 12px;
  }}
  
  .cq-dark .cq-welcome-message {{
    background: #1a237e;
    color: #90caf9;
  }}
  
  .cq-message {{
    display: flex;
    gap: 8px;
    align-items: flex-start;
  }}
  
  .cq-message.cq-user {{
    flex-direction: row-reverse;
  }}
  
  .cq-avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }}
  
  .cq-bubble {{
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 18px;
    line-height: 1.4;
    word-wrap: break-word;
  }}
  
  .cq-user .cq-bubble {{
    background: {primary_color};
    color: white;
  }}
  
  .cq-assistant .cq-bubble {{
    background: #f5f5f5;
    color: #333;
  }}
  
  .cq-dark .cq-assistant .cq-bubble {{
    background: #404040;
    color: white;
  }}
  
  #cq-typing {{
    padding: 16px;
    display: flex;
    justify-content: center;
    gap: 4px;
  }}
  
  .cq-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #ccc;
    animation: cq-bounce 1.4s infinite ease-in-out both;
  }}
  
  .cq-dot:nth-child(1) {{ animation-delay: -0.32s; }}
  .cq-dot:nth-child(2) {{ animation-delay: -0.16s; }}
  
  @keyframes cq-bounce {{
    0%, 80%, 100% {{ transform: scale(0); }}
    40% {{ transform: scale(1); }}
  }}
  
  #cq-footer {{
    padding: 16px;
    border-top: 1px solid #e0e0e0;
  }}
  
  .cq-dark #cq-footer {{
    border-top-color: #404040;
  }}
  
  #cq-attachments {{
    margin-bottom: 8px;
  }}
  
  .cq-attachment {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f0f0f0;
    padding: 4px 8px;
    border-radius: 16px;
    font-size: 12px;
    margin-right: 8px;
    margin-bottom: 4px;
  }}
  
  .cq-dark .cq-attachment {{
    background: #555;
    color: white;
  }}
  
  .cq-attachment button {{
    background: none;
    border: none;
    cursor: pointer;
    padding: 0;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  
  #cq-input {{
    width: 100%;
    min-height: 20px;
    max-height: 100px;
    padding: 12px;
    border: 1px solid #e0e0e0;
    border-radius: 24px;
    resize: none;
    outline: none;
    font-family: inherit;
    font-size: 14px;
    line-height: 1.4;
  }}
  
  .cq-dark #cq-input {{
    background: #404040;
    border-color: #555;
    color: white;
  }}
  
  .cq-actions {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 8px;
  }}
  
  .cq-actions button {{
    background: none;
    border: none;
    cursor: pointer;
    padding: 8px;
    border-radius: 50%;
    transition: background 0.2s;
  }}
  
  .cq-actions button:hover {{
    background: rgba(0,0,0,0.1);
  }}
  
  .cq-dark .cq-actions button:hover {{
    background: rgba(255,255,255,0.1);
  }}
  
  #cq-send {{
    background: {primary_color} !important;
    color: white !important;
  }}
  
  #cq-settings-panel {{
    position: absolute;
    top: 60px;
    right: 0;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    min-width: 200px;
    max-height: 300px;
    overflow-y: auto;
  }}
  
  .cq-dark #cq-settings-panel {{
    background: #2d2d2d;
    border-color: #404040;
    color: white;
  }}
  
  #cq-settings-panel h4 {{
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
  }}
  
  #cq-settings-panel label {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    cursor: pointer;
    font-size: 14px;
  }}
  
  #cq-sources ul {{
    list-style: none;
    padding: 0;
    margin: 8px 0 0 0;
  }}
  
  #cq-sources li {{
    margin-bottom: 4px;
  }}
  
  #cq-sources a {{
    color: {primary_color};
    text-decoration: none;
    font-size: 12px;
  }}
  
  #cq-sources a:hover {{
    text-decoration: underline;
  }}

  .cq-cite-num {{
    margin-left: 4px;
    font-size: 12px;
    text-decoration: none;
  }}

  .cq-cite-window {{
    margin-top: 4px;
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .cq-cite-window span {
    cursor: pointer;
    user-select: none;
  }
  
  .cq-dark #cq-sources a {{
    color: #4fc3f7;
  }}
  
  /* Mobile responsive */
  @media (max-width: 480px) {{
    #cq-chat-widget {{
      width: calc(100vw - 40px) !important;
      height: calc(100vh - 40px) !important;
      max-height: 600px;
    }}
  }}
"""