/* ===================================================
   MIZO 3.0 — ADMIN DASHBOARD & ESP32 SIMULATOR
   =================================================== */

let activeProvider = "groq";
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initNavigation();
  loadAllData();
  setupEventListeners();
  setupAudioBench();
});

// -------------------------------------------------------------
// Live Clock
// -------------------------------------------------------------
function initClock() {
  const clockEl = document.getElementById("live-time");
  const update = () => {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString();
  };
  update();
  setInterval(update, 1000);
}

// -------------------------------------------------------------
// Navigation Tabs
// -------------------------------------------------------------
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const panes = document.querySelectorAll(".tab-pane");
  const titleEl = document.getElementById("page-title");

  const titles = {
    "tab-overview": "System Overview",
    "tab-simulator": "ESP32-S3 Live Simulator",
    "tab-behavior": "AI Rules & Behavior",
    "tab-providers": "API Keys & AI Engine",
    "tab-students": "Student Profiles & Memory",
    "tab-history": "Conversation Logs & Audio",
    "tab-knowledge": "Knowledge Base (RAG)",
    "tab-devices": "Authorized ESP32 Devices"
  };

  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const tabId = item.getAttribute("data-tab");
      navItems.forEach(n => n.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      item.classList.add("active");
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add("active");

      titleEl.textContent = titles[tabId] || "Dashboard";

      // Refresh specific tab data if needed
      if (tabId === "tab-history") loadConversations();
      if (tabId === "tab-knowledge") loadKnowledgeDocs();
      if (tabId === "tab-devices") loadDevices();
    });
  });
}

// -------------------------------------------------------------
// Data Fetching & Sync
// -------------------------------------------------------------
async function loadAllData() {
  await Promise.all([
    loadSettings(),
    loadStudents(),
    loadConversations(),
    loadKnowledgeDocs(),
    loadDevices()
  ]);
}

async function loadSettings() {
  try {
    const res = await fetch("/api/v1/admin/settings");
    if (!res.ok) return;
    const data = await res.json();

    // Populate Overview Stats
    activeProvider = data.active_provider || "groq";
    document.getElementById("stat-active-provider").textContent = formatProviderName(activeProvider);
    document.getElementById("stat-active-model").textContent = data[`${activeProvider}_model`] || "standard";
    document.getElementById("footer-active-provider").textContent = `Provider: ${formatProviderName(activeProvider)}`;

    // Set Provider Cards Active
    document.querySelectorAll(".provider-card").forEach(c => {
      c.classList.toggle("active", c.getAttribute("data-provider") === activeProvider);
    });

    // Populate Behavior Tab
    document.getElementById("setting-system-prompt").value = data.system_prompt || "";
    document.getElementById("setting-coaching-mode").value = data.coaching_mode || "coach";
    document.getElementById("setting-grammar-strictness").value = data.grammar_strictness || "balanced";
    document.getElementById("setting-tts-voice").value = data.tts_voice || "en-US-GuyNeural";
    document.getElementById("setting-tts-rate").value = data.tts_rate || "+0%";

    // Populate Models & Keys
    document.getElementById("input-groq-model").value = data.groq_model || "llama-3.3-70b-versatile";
    document.getElementById("input-openai-model").value = data.openai_model || "gpt-4o-mini";
    document.getElementById("input-ollama-url").value = data.ollama_base_url || "http://localhost:11434";

    document.getElementById("groq-key-status").textContent = `Status: ${data.groq_api_key_masked || "Not set"}`;
    document.getElementById("openai-key-status").textContent = `Status: ${data.openai_api_key_masked || "Not set"}`;
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

function formatProviderName(p) {
  const map = { groq: "Groq Cloud", openai: "OpenAI", qwen: "Qwen / OpenRouter", ollama: "Local Ollama" };
  return map[p] || p;
}

async function loadStudents() {
  try {
    const res = await fetch("/api/v1/students");
    if (!res.ok) return;
    const students = await res.json();

    document.getElementById("stat-student-count").textContent = students.length;

    if (students.length > 0) {
      const s = students[0];
      document.getElementById("stat-active-student-name").textContent = s.name;
      document.getElementById("snap-student-name").textContent = s.name;
      document.getElementById("snap-student-grade").textContent = s.grade;
      document.getElementById("snap-student-goals").textContent = s.learning_goals;

      // Progress bars
      updateBar("fluency", s.fluency_score);
      updateBar("grammar", s.grammar_score);
      updateBar("vocab", s.vocabulary_score);
      updateBar("conf", s.confidence_score);

      // Student grid cards
      const container = document.getElementById("students-container");
      container.innerHTML = students.map(st => `
        <div class="student-card">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
            <div>
              <h3>${escapeHtml(st.name)}</h3>
              <small class="text-muted">${escapeHtml(st.grade)} • ${escapeHtml(st.target_level)}</small>
            </div>
            <span class="tag tag-accent">${st.total_sessions} Sessions</span>
          </div>
          <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:14px;"><strong>Interests:</strong> ${escapeHtml(st.interests)}</p>
          <div class="progress-list">
            <div class="progress-item">
              <div class="progress-labels"><span>Fluency</span><strong>${st.fluency_score}%</strong></div>
              <div class="progress-bar"><div class="progress-fill fill-cyan" style="width:${st.fluency_score}%"></div></div>
            </div>
            <div class="progress-item">
              <div class="progress-labels"><span>Grammar Accuracy</span><strong>${st.grammar_score}%</strong></div>
              <div class="progress-bar"><div class="progress-fill fill-indigo" style="width:${st.grammar_score}%"></div></div>
            </div>
            <div class="progress-item">
              <div class="progress-labels"><span>Vocabulary</span><strong>${st.vocabulary_score}%</strong></div>
              <div class="progress-bar"><div class="progress-fill fill-emerald" style="width:${st.vocabulary_score}%"></div></div>
            </div>
          </div>
        </div>
      `).join("");
    }
  } catch (err) {
    console.error("Error loading students:", err);
  }
}

function updateBar(key, val) {
  val = Math.round(val || 75);
  const textEl = document.getElementById(`snap-${key}`);
  const barEl = document.getElementById(`bar-${key}`);
  if (textEl) textEl.textContent = `${val}%`;
  if (barEl) barEl.style.width = `${val}%`;
}

async function loadConversations() {
  try {
    const res = await fetch("/api/v1/conversations?limit=25");
    if (!res.ok) return;
    const logs = await res.json();
    const box = document.getElementById("history-stream-box");

    if (logs.length === 0) {
      box.innerHTML = `<div class="text-muted" style="text-align:center; padding:30px;">No conversation messages yet. Speak or type to start!</div>`;
      return;
    }

    box.innerHTML = logs.map(msg => {
      const isUser = msg.role === "user";
      const audioBtn = msg.audio_path
        ? `<button class="btn btn-secondary btn-sm" onclick="playAudio('${msg.audio_path}')" style="margin-top:8px;">▶ Listen Speech</button>`
        : "";
      const corrections = (msg.grammar_errors && msg.grammar_errors.length > 0)
        ? `<div style="margin-top:8px; font-size:0.8rem; color:#fbbf24;">
             <strong>Corrections:</strong> ${msg.grammar_errors.map(e => `${escapeHtml(e.error)} ➔ ${escapeHtml(e.correction)}`).join(", ")}
           </div>`
        : "";

      return `
        <div class="history-item ${isUser ? 'history-user' : 'history-assistant'}">
          <div class="history-header">
            <strong>${isUser ? '👤 Student (Voice/Text)' : '🤖 Mikaza AI Spoken Reply'}</strong>
            <span>${new Date(msg.timestamp).toLocaleTimeString()}</span>
          </div>
          <div class="history-content">${escapeHtml(msg.content)}</div>
          ${corrections}
          ${audioBtn}
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

function playAudio(url) {
  const audio = new Audio(url);
  audio.play();
}

async function loadKnowledgeDocs() {
  try {
    const res = await fetch("/api/v1/knowledge/documents");
    if (!res.ok) return;
    const docs = await res.json();
    const container = document.getElementById("doc-list-container");
    document.getElementById("doc-count-tag").textContent = `${docs.length} Documents`;

    if (docs.length === 0) {
      container.innerHTML = `<li class="empty-text text-muted" style="padding:20px; text-align:center;">No documents indexed yet. Drag & drop a PDF textbook above.</li>`;
      return;
    }

    container.innerHTML = docs.map(d => `
      <li class="doc-item">
        <div>
          <strong>📄 ${escapeHtml(d.title)}</strong>
          <div style="font-size:0.75rem; color:var(--text-dim); margin-top:2px;">
            ${d.chunks_count} knowledge chunks • ${new Date(d.created_at).toLocaleDateString()}
          </div>
        </div>
        <button class="btn btn-secondary btn-sm" onclick="deleteDoc(${d.id})" style="color:#f43f5e;">Delete</button>
      </li>
    `).join("");
  } catch (err) {
    console.error("Error loading knowledge docs:", err);
  }
}

async function deleteDoc(docId) {
  if (!confirm("Remove this document from the knowledge base?")) return;
  try {
    await fetch(`/api/v1/knowledge/documents/${docId}`, { method: "DELETE" });
    showToast("Document deleted");
    loadKnowledgeDocs();
  } catch (e) {
    showToast("Delete failed", true);
  }
}

async function loadDevices() {
  try {
    const res = await fetch("/api/v1/devices");
    if (!res.ok) return;
    const devices = await res.json();
    const tbody = document.getElementById("devices-table-body");

    tbody.innerHTML = devices.map(d => `
      <tr>
        <td><strong>${escapeHtml(d.device_name)}</strong></td>
        <td><code>${escapeHtml(d.api_key)}</code></td>
        <td><span class="badge ${d.is_active ? 'badge-accent' : ''}">${d.is_active ? 'Active' : 'Disabled'}</span></td>
        <td>${d.last_seen_at ? new Date(d.last_seen_at).toLocaleTimeString() : 'Never'}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="removeDevice(${d.id})" style="color:#f43f5e;">Revoke</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Error loading devices:", err);
  }
}

async function removeDevice(id) {
  if (!confirm("Revoke this ESP32 authorization key?")) return;
  await fetch(`/api/v1/devices/${id}`, { method: "DELETE" });
  showToast("Device key revoked");
  loadDevices();
}

// -------------------------------------------------------------
// Event Listeners & Actions
// -------------------------------------------------------------
function setupEventListeners() {
  // Provider Selection Cards
  document.querySelectorAll(".provider-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".provider-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      activeProvider = card.getAttribute("data-provider");
    });
  });

  // Toggle Password Visibility
  document.querySelectorAll(".btn-toggle-key").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      input.type = input.type === "password" ? "text" : "password";
      btn.textContent = input.type === "password" ? "👁️" : "🙈";
    });
  });

  // Save AI Behavior
  document.getElementById("btn-save-behavior").addEventListener("click", async () => {
    const payload = {
      system_prompt: document.getElementById("setting-system-prompt").value,
      coaching_mode: document.getElementById("setting-coaching-mode").value,
      grammar_strictness: document.getElementById("setting-grammar-strictness").value,
      tts_voice: document.getElementById("setting-tts-voice").value,
      tts_rate: document.getElementById("setting-tts-rate").value,
    };
    await saveSettingsAPI(payload);
  });

  // Save AI Providers & Keys
  document.getElementById("btn-save-providers").addEventListener("click", async () => {
    const payload = {
      active_provider: activeProvider,
      groq_model: document.getElementById("input-groq-model").value,
      openai_model: document.getElementById("input-openai-model").value,
      ollama_base_url: document.getElementById("input-ollama-url").value,
    };
    const groqKey = document.getElementById("input-groq-key").value.trim();
    const openaiKey = document.getElementById("input-openai-key").value.trim();
    const qwenKey = document.getElementById("input-qwen-key").value.trim();

    if (groqKey) payload.groq_api_key = groqKey;
    if (openaiKey) payload.openai_api_key = openaiKey;
    if (qwenKey) payload.qwen_api_key = qwenKey;

    await saveSettingsAPI(payload);
  });

  // Refresh data button
  document.getElementById("btn-refresh-data").addEventListener("click", () => {
    loadAllData();
    showToast("Data refreshed");
  });

  // Refresh history
  document.getElementById("btn-refresh-history").addEventListener("click", () => {
    loadConversations();
    showToast("History refreshed");
  });

  // Add Device
  document.getElementById("btn-add-device").addEventListener("click", async () => {
    const name = prompt("Enter device name (e.g., 'Classroom Robot 2'):", "ESP32-S3 Node");
    if (!name) return;
    await fetch("/api/v1/devices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_name: name })
    });
    showToast("Device registered!");
    loadDevices();
  });

  // PDF Dropzone
  const dropzone = document.getElementById("pdf-dropzone");
  const fileInput = document.getElementById("pdf-file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.style.borderColor = "#6366f1"; });
  dropzone.addEventListener("dragleave", () => { dropzone.style.borderColor = "rgba(255,255,255,0.15)"; });
  dropzone.addEventListener("drop", e => {
    e.preventDefault();
    dropzone.style.borderColor = "rgba(255,255,255,0.15)";
    if (e.dataTransfer.files.length > 0) uploadDocument(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) uploadDocument(fileInput.files[0]);
  });
}

async function uploadDocument(file) {
  const statusEl = document.getElementById("upload-status");
  statusEl.innerHTML = `<span style="color:#22d3ee;">Uploading and indexing '${escapeHtml(file.name)}'...</span>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/v1/knowledge/upload", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      statusEl.innerHTML = `<span style="color:#34d399;">✓ ${escapeHtml(data.message)}</span>`;
      showToast("Document indexed!");
      loadKnowledgeDocs();
    } else {
      statusEl.innerHTML = `<span style="color:#f43f5e;">✗ ${escapeHtml(data.detail || 'Upload failed')}</span>`;
    }
  } catch (err) {
    statusEl.innerHTML = `<span style="color:#f43f5e;">✗ Upload error: ${err.message}</span>`;
  }
}

async function saveSettingsAPI(payload) {
  try {
    const res = await fetch("/api/v1/admin/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      showToast("Settings saved successfully!");
      loadSettings();
    } else {
      showToast("Error saving settings", true);
    }
  } catch (e) {
    showToast("Network error saving settings", true);
  }
}

// -------------------------------------------------------------
// ESP32 Audio Simulator & Bench
// -------------------------------------------------------------
function setupAudioBench() {
  const btnRecord = document.getElementById("btn-mic-record");
  const recordBtnText = document.getElementById("record-btn-text");
  const recordStatus = document.getElementById("record-status-text");
  const textInput = document.getElementById("sim-text-input");
  const btnSend = document.getElementById("btn-sim-send");

  // Voice recording
  btnRecord.addEventListener("click", async () => {
    if (!isRecording) {
      // Start Recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => {
          if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          sendAudioToBackend(audioBlob);
          // Stop mic tracks
          stream.getTracks().forEach(t => t.stop());
        };

        mediaRecorder.start();
        isRecording = true;
        btnRecord.classList.add("recording");
        recordBtnText.textContent = "Stop & Send";
        recordStatus.textContent = "🔴 Listening... Speak clearly";
      } catch (err) {
        alert("Microphone access denied or not available. Please allow mic permission or use the text input below.");
      }
    } else {
      // Stop Recording
      mediaRecorder.stop();
      isRecording = false;
      btnRecord.classList.remove("recording");
      recordBtnText.textContent = "Hold to Talk";
      recordStatus.textContent = "Processing audio through Mizo cloud...";
    }
  });

  // Text message send
  btnSend.addEventListener("click", () => sendTextMessage());
  textInput.addEventListener("keypress", e => {
    if (e.key === "Enter") sendTextMessage();
  });
}

async function sendAudioToBackend(audioBlob) {
  const recordStatus = document.getElementById("record-status-text");
  const mode = document.getElementById("sim-mode-select").value;

  const formData = new FormData();
  formData.append("audio", audioBlob, "inmp441_audio.wav");
  formData.append("student_id", "1");
  formData.append("session_id", "sim_session");
  formData.append("mode", mode);

  try {
    recordStatus.textContent = "⚡ Transcribing & generating speech...";
    const res = await fetch("/api/v1/esp32/audio?format=json", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    recordStatus.textContent = "Click to start recording voice";
    handleInteractionResponse(data);
  } catch (err) {
    recordStatus.textContent = "Error processing audio";
    console.error(err);
  }
}

async function sendTextMessage() {
  const textInput = document.getElementById("sim-text-input");
  const msg = textInput.value.trim();
  if (!msg) return;
  textInput.value = "";

  const mode = document.getElementById("sim-mode-select").value;

  try {
    const res = await fetch("/api/v1/esp32/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        student_id: 1,
        mode: mode
      })
    });
    const data = await res.json();
    handleInteractionResponse(data);
  } catch (err) {
    console.error("Chat error:", err);
  }
}

function handleInteractionResponse(data) {
  // Display transcription and AI reply
  document.getElementById("sim-result-box").style.display = "flex";
  document.getElementById("sim-transcript-display").textContent = `"${data.transcript}"`;
  document.getElementById("sim-reply-display").textContent = `"${data.reply_text}"`;

  // Display Corrections if any
  const correctionsBox = document.getElementById("sim-corrections-box");
  const correctionsList = document.getElementById("sim-corrections-list");
  if (data.corrections && data.corrections.length > 0) {
    correctionsBox.style.display = "block";
    correctionsList.innerHTML = data.corrections.map(c => `<li>• ${escapeHtml(c)}</li>`).join("");
  } else {
    correctionsBox.style.display = "none";
  }

  // Play Audio through MAX98357A Virtual Player
  if (data.audio_url) {
    const player = document.getElementById("sim-audio-player");
    const noAudioText = document.getElementById("sim-no-audio-text");
    const waveform = document.getElementById("sim-waveform");

    player.src = data.audio_url;
    player.style.display = "block";
    noAudioText.style.display = "none";

    waveform.classList.add("playing");
    player.play();

    player.onended = () => {
      waveform.classList.remove("playing");
    };
  }

  // Update learner snapshot
  if (data.student_stats) {
    updateBar("grammar", data.student_stats.grammar_score);
    updateBar("vocab", data.student_stats.vocabulary_score);
    updateBar("fluency", data.student_stats.fluency_score);
    updateBar("conf", data.student_stats.confidence_score);
  }

  // Reload history
  loadConversations();
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------
function showToast(msg, isError = false) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.style.background = isError ? "rgba(244, 63, 94, 0.95)" : "rgba(16, 185, 129, 0.95)";
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 3000);
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
