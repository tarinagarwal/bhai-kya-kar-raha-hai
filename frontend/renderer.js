const { ipcRenderer } = require("electron");

// Window controls
function minimizeWindow() {
  ipcRenderer.send("window-minimize");
}
function maximizeWindow() {
  ipcRenderer.send("window-maximize");
}
function closeWindow() {
  ipcRenderer.send("window-close");
}

// Markdown config
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang))
      return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
  gfm: true,
});

// State
let ws = null;
let isRunning = false;
let actionCountNum = 0;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// DOM
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const voiceBtn = document.getElementById("voiceBtn");
const screenshot = document.getElementById("screenshot");
const placeholder = document.getElementById("placeholder");
const browserStatus = document.getElementById("browserStatus");
const urlText = document.getElementById("urlText");
const thinkingContent = document.getElementById("thinkingContent");
const thinkingIndicator = document.getElementById("thinkingIndicator");
const actionsList = document.getElementById("actionsList");
const actionCount = document.getElementById("actionCount");
const connectionDot = document.getElementById("connectionDot");
const connectionStatus = document.getElementById("connectionStatus");
const iterationCount = document.getElementById("iterationCount");
const modelSelect = document.getElementById("modelSelect");
const modeSelect = document.getElementById("modeSelect");
const modeIndicator = document.getElementById("modeIndicator");
const cursorIndicator = document.getElementById("cursorIndicator");
const memoryStatus = document.getElementById("memoryStatus");

// ── WebSocket ──

function connectWebSocket() {
  ws = new WebSocket("ws://localhost:8765");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    connectionDot.classList.add("connected");
    connectionStatus.textContent = "Connected";
  };
  ws.onclose = () => {
    connectionDot.classList.remove("connected");
    connectionStatus.textContent = "Disconnected";
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = () => {
    connectionDot.classList.remove("connected");
    connectionStatus.textContent = "Connection error";
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleMessage(JSON.parse(event.data));
    }
  };
}

function handleMessage(data) {
  switch (data.type) {
    case "screenshot":
    case "live_frame":
      updateScreenshot(data.image, data.url);
      break;
    case "thinking":
      thinkingIndicator.classList.remove("hidden");
      addThinkingEntry(data.text);
      break;
    case "action":
      addActionItem(data.name, data.args);
      showCursorAt(data.args);
      break;
    case "iteration":
      iterationCount.textContent = `Iteration: ${data.count}`;
      break;
    case "complete":
      thinkingIndicator.classList.add("hidden");
      setRunning(false);
      browserStatus.textContent = "✓ Complete";
      if (data.result) addThinkingEntry(data.result, "result");
      break;
    case "error":
      setRunning(false);
      browserStatus.textContent = "✕ Error";
      addThinkingEntry(data.message, "error");
      break;
    case "transcription":
      handleTranscription(data);
      break;
    case "app_list":
      handleAppList(data.apps);
      break;
  }
}

function updateScreenshot(imageB64, url) {
  screenshot.src = "data:image/png;base64," + imageB64;
  screenshot.classList.remove("hidden");
  placeholder.classList.add("hidden");
  if (url) urlText.textContent = url;
}

function showCursorAt(args) {
  if (!args || (!args.x && args.x !== 0)) return;
  const viewport = document.getElementById("browserViewport");
  const rect = viewport.getBoundingClientRect();
  const px = (args.x / 1000) * rect.width;
  const py = (args.y / 1000) * rect.height;
  cursorIndicator.style.left = px + "px";
  cursorIndicator.style.top = py + "px";
  cursorIndicator.classList.remove("hidden");
  clearTimeout(cursorIndicator._hideTimer);
  cursorIndicator._hideTimer = setTimeout(
    () => cursorIndicator.classList.add("hidden"),
    2000,
  );
}

// ── Voice Input ──

async function toggleVoice() {
  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 48000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: "audio/webm;codecs=opus",
    });

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(audioChunks, { type: "audio/webm;codecs=opus" });
      const buffer = await blob.arrayBuffer();

      // Send as base64 via JSON
      const b64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "voice_data", audio: b64 }));
        browserStatus.textContent = "Transcribing...";
      }
    };

    mediaRecorder.start();
    isRecording = true;
    voiceBtn.classList.add("recording");
    browserStatus.textContent = "🎙️ Recording...";
  } catch (e) {
    console.error("Mic error:", e);
    browserStatus.textContent = "Mic access denied";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  voiceBtn.classList.remove("recording");
}

function handleTranscription(data) {
  if (data.text) {
    queryInput.value = data.text;
    browserStatus.textContent = "Transcribed";
    // Auto-submit if we got text
    if (data.text.trim()) {
      startAgent();
    }
  } else {
    browserStatus.textContent = data.error || "No speech detected";
  }
}

function handleAppList(apps) {
  if (apps && apps.length > 0) {
    const list = apps.map((a) => `${a.process}: ${a.title}`).join("\n");
    addThinkingEntry(`**Open Apps:**\n\`\`\`\n${list}\n\`\`\``, "result");
  }
}

// ── UI Helpers ──

function renderMarkdown(text) {
  try {
    return marked.parse(text);
  } catch (e) {
    return escapeHtml(text);
  }
}

function addThinkingEntry(text, label = "thinking") {
  if (!text) return;
  const empty = thinkingContent.querySelector(".empty-state");
  if (empty) empty.remove();

  const entry = document.createElement("div");
  entry.className = "thinking-entry";
  const labelClass = `label-${label}`;
  const labelText = label.charAt(0).toUpperCase() + label.slice(1);

  entry.innerHTML = `
    <div class="thinking-label ${labelClass}">${labelText}</div>
    <div class="md-body">${renderMarkdown(text)}</div>
  `;
  entry
    .querySelectorAll("pre code")
    .forEach((block) => hljs.highlightElement(block));
  thinkingContent.appendChild(entry);
  thinkingContent.scrollTop = thinkingContent.scrollHeight;
}

function addActionItem(name, args) {
  const empty = actionsList.querySelector(".empty-state");
  if (empty) empty.remove();
  actionCountNum++;
  actionCount.textContent = actionCountNum;

  const icons = {
    click_at: "🖱️",
    type_text_at: "⌨️",
    scroll_document: "📜",
    scroll_at: "📜",
    navigate: "🌐",
    go_back: "⬅️",
    go_forward: "➡️",
    search: "🔍",
    wait_5_seconds: "⏳",
    hover_at: "👆",
    key_combination: "⌨️",
    drag_and_drop: "✋",
    open_web_browser: "🌐",
    open_app: "🚀",
    close_app: "❌",
    switch_to_app: "🔄",
    list_open_apps: "📋",
    run_shell_command: "💻",
    focus_window: "🪟",
    calculator_compute: "🔢",
    desktop_action: "🖥️",
  };
  const time = new Date().toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const argsStr = args
    ? Object.entries(args)
        .filter(([k]) => k !== "safety_decision")
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ")
    : "";

  const item = document.createElement("div");
  item.className = "action-item";
  item.innerHTML = `
    <div class="action-icon">${icons[name] || "⚡"}</div>
    <div class="action-details">
      <div class="action-name">${name}</div>
      ${argsStr ? `<div class="action-args">${escapeHtml(argsStr)}</div>` : ""}
    </div>
    <div class="action-time">${time}</div>
  `;
  actionsList.appendChild(item);
  actionsList.scrollTop = actionsList.scrollHeight;
}

function startAgent() {
  if (isRunning || !ws || ws.readyState !== WebSocket.OPEN) return;
  const query = queryInput.value.trim();
  if (!query) return;
  setRunning(true);
  thinkingContent.innerHTML = "";
  actionsList.innerHTML = "";
  actionCountNum = 0;
  actionCount.textContent = "0";
  iterationCount.textContent = "Iteration: 0";
  ws.send(
    JSON.stringify({
      type: "start",
      query: query,
      model: modelSelect.value,
      mode: modeSelect.value,
    }),
  );
}

function setRunning(running) {
  isRunning = running;
  sendBtn.disabled = running;
  queryInput.disabled = running;
  if (running) {
    thinkingIndicator.classList.remove("hidden");
    browserStatus.textContent = "Starting...";
  } else {
    thinkingIndicator.classList.add("hidden");
  }
}

function updateMode() {
  const mode = modeSelect.value;
  if (mode === "desktop") {
    modeIndicator.textContent = "🖥️ Desktop (clawd-cursor)";
  } else {
    modeIndicator.textContent = "🌐 Browser (Playwright)";
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Init
connectWebSocket();
