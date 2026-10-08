/* ===================================================
   MIKAZA / MIZO 3.0 — SOFTWARE ONBOARDING & DASHBOARD
   WITH TWO-WAY CONTINUOUS VOICE CONVERSATION (VAD)
   =================================================== */

let activeProvider = "groq";
let activeSessionId = null;
let currentStudentId = 1;

// Audio & Recording State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordStartTime = null;
let recordTimerInterval = null;
let lastRecordedBlob = null;
let lastRecordedDuration = 0;

// Two-Way Continuous Voice Mode & State Machine
const VoiceState = {
  WAKE_LISTENING: "WAKE_LISTENING",
  ACTIVE_CONVERSATION: "ACTIVE_CONVERSATION",
  SNOOZE: "STANDBY",
  SNOOZED: "STANDBY",
  STANDBY: "STANDBY",
  WAKE_ONLY: "STANDBY",
  WAKE_DETECTED: "WAKE_DETECTED",
  ONBOARDING: "ONBOARDING",
  ASSESSMENT: "ASSESSMENT",
  PROCESSING_ASSESSMENT: "PROCESSING_ASSESSMENT",
  ASSESSMENT_COMPLETE: "ASSESSMENT_COMPLETE",
  READY: "READY",
  CONVERSATION: "CONVERSATION",
  AWAKE: "ACTIVE_LISTENING",
  ACTIVE_LISTENING: "ACTIVE_LISTENING",
  CAPTURING: "CAPTURING",
  PROCESSING: "PROCESSING",
  SPEAKING: "SPEAKING",
  STOPPING: "STOPPING",
  ERROR_RECOVERY: "ERROR_RECOVERY",
  ASSESSMENT_RECORDING: "ASSESSMENT_RECORDING"
};

let currentVoiceState = VoiceState.WAKE_LISTENING;
let isTwoWayVoiceMode = true; // Auto-listen active by default after speech
let isMicMuted = false;
let audioContext = null;
let analyserNode = null;
let micMediaStream = null;
let vadAnimationId = null;
let speechDetected = false;
let silenceStartTimestamp = null;
let isMizoSpeaking = false;
let isMikazaSpeaking = false;
let isProcessingBackend = false;
let ambientNoiseFloor = 0.005;
let calibrationFrames = 0;
let wakeWordRecognizer = null;
let isWakeWordEnabled = true;

// Separate, clearly defined timers
let turnEndingSilenceSeconds = 3.0; // Normal conversation turn pause threshold (3s)
let inactivityTimeoutSeconds = 10.0; // 10-second inactivity = SNOOZE / STANDBY, NOT microphone shutdown
let inactivityTimerId = null;
let currentAbortController = null;

// Dedicated Assessment Recording state
let assessmentStartTime = null;
let assessmentTimerInterval = null;
let assessmentSpeechDetected = false;
let assessmentSilenceStart = null;
let assessmentSilenceTimeoutSeconds = 3.0;
let isAssessmentFinalizing = false;

document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initNavigation();
  loadAllData();
  setupEventListeners();
  setupAudioBench();
  initWakeWordListener();
  setupSettingsEventHandlers();
});

// -------------------------------------------------------------
// Live Clock
// -------------------------------------------------------------
function initClock() {
  const clockEl = document.getElementById("live-time");
  if (!clockEl) return;
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
    "tab-simulator": "Normal Chat & English Coach",
    "tab-study": "Study & Material-Based Teaching",
    "tab-speech": "Speech & Seminar Practice",
    "tab-overview": "System Overview & Status",
    "tab-behavior": "AI Rules & Behavior",
    "tab-providers": "AI Provider Cascade & API Keys",
    "tab-students": "Learner Profile & Baseline",
    "tab-history": "Conversation History",
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

      if (titleEl) titleEl.textContent = titles[tabId] || "Dashboard";

      // Refresh specific tab data if needed
      if (tabId === "tab-history") loadConversations();
      if (tabId === "tab-knowledge") loadKnowledgeDocs();
      if (tabId === "tab-devices") loadDevices();
      if (tabId === "tab-students") { loadStudents(); loadStudentMistakes(); loadAssessments(); }
      if (tabId === "tab-overview") { loadProviderStatus(); loadActiveSession(); }
      if (tabId === "tab-simulator") { loadStudents(); loadConversations(); }
      if (tabId === "tab-study") loadStudyWorkspace();
      if (tabId === "tab-speech") loadSpeechWorkspace();
    });
  });
}

// -------------------------------------------------------------
// Data Fetching & Sync
// -------------------------------------------------------------
async function loadAllData() {
  await Promise.all([
    loadSettings(),
    loadProviderStatus(),
    loadStudents(),
    loadHistoricalProgress(),
    loadActiveSession(),
    loadConversations(),
    loadKnowledgeDocs(),
    loadDevices(),
    loadStudentMistakes(),
    loadAssessments()
  ]);
}

async function loadSettings() {
  try {
    const res = await fetch("/api/v1/admin/settings");
    if (!res.ok) return;
    const data = await res.json();

    activeProvider = data.active_provider || "groq";
    const statProv = document.getElementById("stat-active-provider");
    if (statProv) statProv.textContent = formatProviderName(activeProvider);
    const statMod = document.getElementById("stat-active-model");
    if (statMod) statMod.textContent = data[`${activeProvider}_model`] || "GPT OSS 120B";
    const footProv = document.getElementById("footer-active-provider");
    if (footProv) footProv.textContent = `Primary: ${formatProviderName(activeProvider)}`;

    // Set Provider Cards Active
    document.querySelectorAll(".provider-card").forEach(c => {
      c.classList.toggle("active", c.getAttribute("data-provider") === activeProvider);
    });

    // Populate Behavior Tab
    const sysPrompt = document.getElementById("setting-system-prompt");
    if (sysPrompt) sysPrompt.value = data.system_prompt || "";
    const coachMode = document.getElementById("setting-coaching-mode");
    if (coachMode) coachMode.value = data.coaching_mode || "coach";
    const gramStrict = document.getElementById("setting-grammar-strictness");
    if (gramStrict) gramStrict.value = data.grammar_strictness || "balanced";
    const ttsVoice = document.getElementById("setting-tts-voice");
    if (ttsVoice) ttsVoice.value = data.tts_voice || "en-US-GuyNeural";
    const ttsRate = document.getElementById("setting-tts-rate");
    if (ttsRate) ttsRate.value = data.tts_rate || "+0%";

    // Populate Models & Keys
    const groqMod = document.getElementById("input-groq-model");
    if (groqMod) groqMod.value = data.groq_model || "openai/gpt-oss-120b";
    const openaiMod = document.getElementById("input-openai-model");
    if (openaiMod) openaiMod.value = data.openai_model || "gpt-4o-mini";
    const qwenMod = document.getElementById("input-qwen-model");
    if (qwenMod) qwenMod.value = data.qwen_model || "qwen/qwen-2.5-72b-instruct";
    const ollamaUrl = document.getElementById("input-ollama-url");
    if (ollamaUrl) ollamaUrl.value = data.ollama_base_url || "http://localhost:11434";
    const ollamaMod = document.getElementById("input-ollama-model");
    if (ollamaMod) ollamaMod.value = data.ollama_model || "llama3:latest";

    // Dynamic catalog refresh for Groq models
    fetchProviderModels("groq");

    // Key status indicators and placeholders
    const groqKeyInput = document.getElementById("input-groq-key");
    const groqKeyStat = document.getElementById("groq-key-status");
    if (groqKeyStat) {
      if (data.groq_api_key_configured) {
        groqKeyStat.innerHTML = `✓ Stored: <code style="color:#34d399;">${escapeHtml(data.groq_api_key_masked)}</code> (leave blank to keep)`;
        if (groqKeyInput && !groqKeyInput.value) groqKeyInput.placeholder = "•••••••••••••••• (leave blank to keep)";
      } else {
        groqKeyStat.textContent = "Status: Not configured";
        if (groqKeyInput) groqKeyInput.placeholder = "gsk_...";
      }
    }

    const openaiKeyInput = document.getElementById("input-openai-key");
    const openaiKeyStat = document.getElementById("openai-key-status");
    if (openaiKeyStat) {
      if (data.openai_api_key_configured) {
        openaiKeyStat.innerHTML = `✓ Stored: <code style="color:#34d399;">${escapeHtml(data.openai_api_key_masked)}</code> (leave blank to keep)`;
        if (openaiKeyInput && !openaiKeyInput.value) openaiKeyInput.placeholder = "•••••••••••••••• (leave blank to keep)";
      } else {
        openaiKeyStat.textContent = "Status: Not configured";
        if (openaiKeyInput) openaiKeyInput.placeholder = "sk-...";
      }
    }

    const qwenKeyInput = document.getElementById("input-qwen-key");
    const qwenKeyStat = document.getElementById("qwen-key-status");
    if (qwenKeyStat) {
      if (data.qwen_api_key_configured) {
        qwenKeyStat.innerHTML = `✓ Stored: <code style="color:#34d399;">${escapeHtml(data.qwen_api_key_masked)}</code> (leave blank to keep)`;
        if (qwenKeyInput && !qwenKeyInput.value) qwenKeyInput.placeholder = "•••••••••••••••• (leave blank to keep)";
      } else {
        qwenKeyStat.textContent = "Status: Not configured";
        if (qwenKeyInput) qwenKeyInput.placeholder = "sk-or-v1-...";
      }
    }
  } catch (err) {
    console.error("Error loading settings:", err);
  }
}

function updateProviderUIStatus(prov, data) {
  if (!prov || !data) return { statusClass: "badge-neutral", statusText: "○ NOT CONFIGURED", pillText: "○ NOT CONFIGURED" };
  const p = prov.toLowerCase();
  const st = (data.status || "").toUpperCase();
  const reason = (data.reason || "").toUpperCase();
  const isConfigured = data.configured !== false && st !== "NOT_CONFIGURED";

  let statusClass = "badge-neutral";
  let statusText = "○ NOT CONFIGURED";
  let pillText = "○ NOT CONFIGURED";
  let pillBg = "rgba(255, 255, 255, 0.08)";
  let pillColor = "var(--text-muted)";
  let keyStatusHtml = "";

  if (st === "READY" || st === "CONNECTED") {
    statusClass = "badge-success";
    statusText = `● CONNECTED (${data.latency_ms || 0}ms)`;
    pillText = "● CONNECTED";
    pillBg = "rgba(16, 185, 129, 0.2)";
    pillColor = "#34d399";
    keyStatusHtml = `<span style="color:#34d399; font-weight:600;">✓ CONNECTED / READY</span> <span style="color:var(--text-muted); font-size:0.75rem;">(${data.latency_ms || 0}ms)</span>`;
  } else if (st === "INVALID_API_KEY" || st === "AUTHENTICATION_ERROR") {
    statusClass = "badge-danger";
    statusText = "● INVALID API KEY";
    pillText = "● INVALID API KEY";
    pillBg = "rgba(244, 63, 94, 0.2)";
    pillColor = "#f87171";
    keyStatusHtml = `<span style="color:#f87171; font-weight:600;">✗ INVALID API KEY</span> <span style="color:var(--text-muted); font-size:0.75rem;">(HTTP 401 authentication failed)</span>`;
  } else if (st === "RATE_LIMITED" || st === "QUOTA_EXCEEDED" || (st === "AUTHENTICATED" && reason === "QUOTA_EXCEEDED")) {
    statusClass = "badge-warning";
    statusText = "● RATE LIMITED / QUOTA EXCEEDED";
    pillText = "● RATE LIMITED";
    pillBg = "rgba(245, 158, 11, 0.2)";
    pillColor = "#fbbf24";
    keyStatusHtml = `<span style="color:#fbbf24; font-weight:600;">⚠️ RATE LIMITED / QUOTA EXCEEDED</span> <span style="color:var(--text-muted); font-size:0.75rem;">(HTTP 429)</span>`;
  } else if (st === "MODEL_UNAVAILABLE" || st === "MODEL_ERROR") {
    statusClass = "badge-warning";
    statusText = "● MODEL UNAVAILABLE";
    pillText = "● MODEL UNAVAILABLE";
    pillBg = "rgba(245, 158, 11, 0.2)";
    pillColor = "#fbbf24";
    keyStatusHtml = `<span style="color:#fbbf24; font-weight:600;">⚠️ MODEL UNAVAILABLE</span> <span style="color:var(--text-muted); font-size:0.75rem;">(${escapeHtml(data.model || "")})</span>`;
  } else if (st === "PERMISSION_DENIED" || st === "FORBIDDEN") {
    statusClass = "badge-danger";
    statusText = "● PERMISSION DENIED (403)";
    pillText = "● PERMISSION DENIED";
    pillBg = "rgba(244, 63, 94, 0.2)";
    pillColor = "#f87171";
    keyStatusHtml = `<span style="color:#f87171; font-weight:600;">✗ PERMISSION DENIED (HTTP 403)</span>`;
  } else if (st === "NOT_CONFIGURED" || !isConfigured) {
    statusClass = "badge-neutral";
    statusText = "○ NOT CONFIGURED";
    pillText = "○ NOT CONFIGURED";
    pillBg = "rgba(255, 255, 255, 0.08)";
    pillColor = "var(--text-muted)";
    keyStatusHtml = `<span style="color:var(--text-muted);">Status: Not configured</span>`;
  } else {
    statusClass = "badge-danger";
    statusText = "● " + (data.display_status || "PROVIDER UNREACHABLE");
    pillText = "● " + (data.display_status || "UNREACHABLE");
    pillBg = "rgba(244, 63, 94, 0.2)";
    pillColor = "#f87171";
    keyStatusHtml = `<span style="color:#f87171; font-weight:600;">✗ ${escapeHtml(data.error || data.details || 'Connection error')}</span>`;
  }

  // 1. Update provider selector card pill on Tab 4
  const card = document.getElementById(`prov-${p}`);
  if (card) {
    let pill = card.querySelector(".card-status-pill");
    if (!pill) {
      pill = document.createElement("div");
      pill.className = "card-status-pill";
      pill.style.cssText = "font-size:0.75rem; font-weight:600; margin-top:8px; display:inline-block; padding:3px 8px; border-radius:4px;";
      card.appendChild(pill);
    }
    pill.style.background = pillBg;
    pill.style.color = pillColor;
    pill.textContent = pillText;
  }

  // 2. Update key status text under input on Tab 4
  const statEl = document.getElementById(`${p}-key-status`);
  if (statEl && keyStatusHtml) {
    statEl.innerHTML = keyStatusHtml;
  }

  // 3. Update overview dashboard strip card if already rendered
  const stripCard = document.querySelector(`.provider-status-card[data-provider="${p}"]`);
  if (stripCard) {
    const badge = stripCard.querySelector(".badge");
    if (badge) {
      badge.className = `badge ${statusClass}`;
      badge.textContent = statusText;
    }
  }

  return { statusClass, statusText, pillText };
}

async function testSingleProvider(provider) {
  const prov = (provider || "").toLowerCase().trim();
  if (!prov) return null;

  const btn = document.querySelector(`.btn-test-single-key[data-provider="${prov}"]`);
  let originalText = "";
  if (btn) {
    originalText = btn.textContent;
    btn.textContent = "⏳...";
    btn.disabled = true;
  }

  try {
    const url = `/api/v1/admin/settings/provider-status/${prov}`;
    const res = await fetch(url);
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || errData.error || `HTTP ${res.status}`);
    }
    const data = await res.json();

    // Immediately update UI for this provider
    updateProviderUIStatus(prov, data);

    const st = (data.status || "").toUpperCase();
    const reason = (data.reason || "").toUpperCase();

    if (st === "READY" || st === "CONNECTED") {
      showToast(`✓ ${formatProviderName(prov)} is CONNECTED & READY! (${data.latency_ms || 0}ms)`);
    } else if (st === "NOT_CONFIGURED") {
      showToast(`○ ${formatProviderName(prov)}: NOT CONFIGURED`, true);
    } else if (st === "INVALID_API_KEY" || st === "AUTHENTICATION_ERROR") {
      showToast(`✗ ${formatProviderName(prov)}: INVALID API KEY (HTTP 401)`, true);
    } else if (st === "RATE_LIMITED" || st === "QUOTA_EXCEEDED" || (st === "AUTHENTICATED" && reason === "QUOTA_EXCEEDED")) {
      showToast(`⚠️ ${formatProviderName(prov)}: QUOTA EXCEEDED / RATE LIMITED (HTTP 429)`, true);
    } else if (st === "MODEL_UNAVAILABLE" || st === "MODEL_ERROR") {
      showToast(`⚠️ ${formatProviderName(prov)}: MODEL UNAVAILABLE (${data.model})`, true);
    } else if (st === "PERMISSION_DENIED" || st === "FORBIDDEN") {
      showToast(`✗ ${formatProviderName(prov)}: PERMISSION DENIED (HTTP 403)`, true);
    } else {
      showToast(`✗ ${formatProviderName(prov)}: ${data.details || data.error || 'Connection failed'}`, true);
    }

    return data;
  } catch (err) {
    showToast(`Failed to test ${formatProviderName(prov)}: ${err.message}`, true);
    return null;
  } finally {
    if (btn) {
      btn.textContent = originalText;
      btn.disabled = false;
    }
  }
}

async function fetchProviderModels(provider = "groq") {
  const p = (provider || "groq").toLowerCase().trim();
  const datalist = document.getElementById(`${p}-model-list`);
  const statusEl = document.getElementById(`${p}-model-status`);
  const inputEl = document.getElementById(`input-${p}-model`);

  if (!datalist) return;

  try {
    const res = await fetch(`/api/v1/admin/providers/${p}/models`);
    if (!res.ok) return;
    const data = await res.json();
    const models = data.models || [];

    if (models.length > 0) {
      datalist.innerHTML = "";
      models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        const tag = m.deprecated ? " (⚠️ DEPRECATED/UNAVAILABLE)" : (m.id.includes("120b") ? " (Flagship Production)" : "");
        opt.textContent = `${m.id}${tag}`;
        datalist.appendChild(opt);
      });

      const currentVal = inputEl ? inputEl.value.trim() : "";
      const currentMeta = models.find(m => m.id === currentVal);
      if ((currentMeta && currentMeta.deprecated) || currentVal === "llama-3.3-70b-versatile") {
        if (statusEl) {
          statusEl.innerHTML = `<span style="color:#fbbf24; font-weight:600;">⚠️ Model ${escapeHtml(currentVal)} was deprecated on Aug 16, 2026.</span> <button type="button" class="btn btn-sm btn-outline-warning" id="btn-auto-switch-groq" style="margin-left:8px; padding:2px 8px; font-size:0.75rem;">Switch to openai/gpt-oss-120b</button>`;
          document.getElementById("btn-auto-switch-groq")?.addEventListener("click", () => {
            if (inputEl) inputEl.value = "openai/gpt-oss-120b";
            statusEl.innerHTML = `<span style="color:#34d399; font-weight:600;">✓ Updated to openai/gpt-oss-120b. Click "Save Provider Settings" below.</span>`;
          });
        }
      } else if (statusEl) {
        statusEl.innerHTML = `<span style="color:var(--text-muted);">Verified active production models loaded from Groq catalog.</span>`;
      }
    }
  } catch (err) {
    console.warn(`[MODELS] Could not fetch dynamic models for ${p}:`, err);
  }
}

async function loadProviderStatus() {
  const container = document.getElementById("provider-status-container");
  if (!container) return;

  try {
    const res = await fetch("/api/v1/admin/settings/provider-status");
    if (!res.ok) return;
    const data = await res.json();

    const providerNames = {
      groq: "Groq",
      openai: "OpenAI",
      qwen: "OpenRouter / Qwen",
      ollama: "Local Ollama"
    };

    container.innerHTML = data.providers.map(p => {
      const { statusClass, statusText } = updateProviderUIStatus(p.provider, p);
      let detailSnippet = "";

      const st = (p.status || "").toUpperCase();
      const reason = (p.reason || "").toUpperCase();

      if (st === "READY" || (st === "CONNECTED" && p.generation_ready !== false)) {
        detailSnippet = `<div style="font-size:0.75rem; color:#10b981; margin-top:3px;">
          ✓ Fully operational & ready for live speech/text generation
        </div>`;
      } else if (st === "AUTHENTICATED" || reason === "QUOTA_EXCEEDED" || st === "QUOTA_EXCEEDED") {
        const errDetail = p.details || p.error || "Generation unavailable: Insufficient quota (HTTP 429)";
        detailSnippet = `<div style="font-size:0.75rem; color:#fbbf24; margin-top:3px;">
          <span style="background:rgba(245,158,11,0.25); color:#fde68a; padding:1px 5px; border-radius:4px; font-weight:600; font-size:0.7rem; margin-right:4px;">AUTHENTICATED</span>
          Generation unavailable: ${escapeHtml(errDetail)}
        </div>`;
      } else if (st === "RATE_LIMITED") {
        detailSnippet = `<div style="font-size:0.75rem; color:#fbbf24; margin-top:3px;">
          <span style="background:rgba(245,158,11,0.25); color:#fde68a; padding:1px 5px; border-radius:4px; font-weight:600; font-size:0.7rem; margin-right:4px;">RATE LIMITED</span>
          Generation temporarily unavailable: Rate limit exceeded (HTTP 429)
        </div>`;
      } else if (st === "INVALID_API_KEY" || st === "AUTHENTICATION_ERROR") {
        detailSnippet = `<div style="font-size:0.75rem; color:#f87171; margin-top:3px;">
          Authentication failed: Invalid API key (HTTP 401)
        </div>`;
      } else if (st === "MODEL_UNAVAILABLE" || st === "MODEL_ERROR") {
        detailSnippet = `<div style="font-size:0.75rem; color:#fbbf24; margin-top:3px;">
          Configured model <code>${escapeHtml(p.model)}</code> is not available for this account. Please select a supported model.
        </div>`;
      } else if (st === "PERMISSION_DENIED" || st === "FORBIDDEN") {
        detailSnippet = `<div style="font-size:0.75rem; color:#f87171; margin-top:3px;">
          Access forbidden: API key does not have permissions for this resource (HTTP 403)
        </div>`;
      } else if (st === "NETWORK_ERROR" || st === "UNAVAILABLE" || st === "TIMEOUT") {
        detailSnippet = `<div style="font-size:0.75rem; color:#f87171; margin-top:3px;">
          Provider unreachable: ${escapeHtml(p.details || p.error || "Connection timed out / server offline")}
        </div>`;
      } else if (st === "NOT_CONFIGURED" || !p.configured) {
        detailSnippet = `<div style="font-size:0.75rem; color:var(--text-muted); margin-top:3px;">
          No API credentials configured
        </div>`;
      } else {
        detailSnippet = `<div style="font-size:0.75rem; color:#f87171; margin-top:3px;">
          ${escapeHtml(p.error || p.details || "Provider error")}
        </div>`;
      }

      return `
        <div class="provider-status-card" data-provider="${p.provider}" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 8px;">
          <div style="flex: 1; padding-right: 12px;">
            <strong>${providerNames[p.provider] || p.provider}</strong>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">
              Configured Model: <code style="color:var(--text-primary); font-size:0.75rem;">${escapeHtml(p.model)}</code>
            </div>
            ${detailSnippet}
          </div>
          <span class="badge ${statusClass}" style="white-space:nowrap; margin-top:2px;">${statusText}</span>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Error checking provider status:", err);
  }
}

async function loadActiveSession() {
  try {
    const res = await fetch(`/api/v1/students/${currentStudentId}/sessions/active`);
    if (!res.ok) return;
    const session = await res.json();

    activeSessionId = session.session_id;
    const statSes = document.getElementById("stat-active-session-id");
    if (statSes) statSes.textContent = session.session_id.substring(0, 10) + "...";
    const statCount = document.getElementById("stat-session-msg-count");
    if (statCount) statCount.textContent = `${session.metrics.message_count} messages in session`;
    const badgeMode = document.getElementById("badge-session-mode");
    if (badgeMode) badgeMode.textContent = `${(session.mode || 'coach').toUpperCase()} MODE`;

    // Populate Current Session Metrics
    const msgCount = session.metrics.message_count || 0;
    updateSessionBar("session-fluency", session.metrics.fluency_score, msgCount);
    updateSessionBar("session-grammar", session.metrics.grammar_accuracy, msgCount);
    updateSessionBar("session-vocab", session.metrics.vocabulary_richness, msgCount);
    updateSessionBar("session-pacing", session.metrics.pacing_score, msgCount);
  } catch (err) {
    console.error("Error loading active session:", err);
  }
}

function updateSessionBar(idPrefix, val, messageCount = 0) {
  const textEl = document.getElementById(`${idPrefix}-text`);
  const barEl = document.getElementById(`${idPrefix}-bar`);
  if (!messageCount || val === null || val === undefined) {
    if (textEl) textEl.textContent = "Not assessed yet";
    if (barEl) barEl.style.width = "0%";
    return;
  }
  const rounded = Math.round(val);
  if (textEl) textEl.textContent = `${rounded}.0%`;
  if (barEl) barEl.style.width = `${rounded}%`;
}

function formatProviderName(p) {
  const map = { groq: "Groq Llama 3.3", openai: "OpenAI GPT-4o", qwen: "Qwen 2.5", ollama: "Local Ollama" };
  return map[p] || p;
}

// -------------------------------------------------------------
// Onboarding Stepper Management
// -------------------------------------------------------------
function updateOnboardingStepper(step, isCompleted) {
  const steps = [
    { id: "step-node-name", divId: "step-div-1", match: ["ask_name", "new", "welcome"] },
    { id: "step-node-goals", divId: "step-div-2", match: ["ask_goals", "ask_problems", "ask_about_user"] },
    { id: "step-node-weaknesses", divId: "step-div-3", match: ["ask_weaknesses", "ask_interests"] },
    { id: "step-node-level", divId: "step-div-4", match: ["ask_self_level"] },
    { id: "step-node-speech", divId: "step-div-5", match: ["speech_test_prompt", "speaking_assessment", "speech_evaluation"] },
    { id: "step-node-complete", divId: null, match: ["completed"] }
  ];

  const badge = document.getElementById("badge-onboarding-step-indicator");
  if (badge) {
    badge.textContent = isCompleted ? "Onboarding: Completed" : `Step: ${(step || 'ASK_NAME').toUpperCase()}`;
    badge.className = isCompleted ? "tag tag-success" : "tag tag-accent";
  }

  let activeIndex = 0;
  if (isCompleted || step === "completed") {
    activeIndex = 5;
  } else {
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].match.includes(step)) {
        activeIndex = i;
        break;
      }
    }
  }

  steps.forEach((s, idx) => {
    const node = document.getElementById(s.id);
    const div = s.divId ? document.getElementById(s.divId) : null;
    if (!node) return;

    node.classList.remove("active", "completed");
    if (div) div.classList.remove("active");

    if (idx < activeIndex || (isCompleted && idx <= 5)) {
      node.classList.add("completed");
      if (div) div.classList.add("active");
    } else if (idx === activeIndex && !isCompleted) {
      node.classList.add("active");
    }
  });

  const skipBtn = document.getElementById("btn-skip-audio-onboarding");
  if (skipBtn) {
    skipBtn.style.display = isCompleted ? "none" : "inline-flex";
  }
}

async function loadStudents() {
  try {
    const res = await fetch("/api/v1/students");
    if (!res.ok) return;
    const students = await res.json();

    if (students.length > 0) {
      const s = students[0];
      currentStudentId = s.id;

      const statName = document.getElementById("stat-active-student-name");
      if (statName) statName.textContent = s.name;
      const statOnboard = document.getElementById("stat-student-onboarding-status");
      if (statOnboard) statOnboard.textContent = s.onboarding_completed ? "Onboarding: Complete" : `Onboarding: ${s.onboarding_step}`;
      const snapGrade = document.getElementById("snap-student-grade");
      if (snapGrade) snapGrade.textContent = s.grade || s.education || "Student";

      updateOnboardingStepper(s.onboarding_step, s.onboarding_completed);

      await loadHistoricalProgress();

      const container = document.getElementById("students-container");
      if (container) {
        container.innerHTML = students.map(st => `
          <div class="student-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
              <div>
                <h3>${escapeHtml(st.name)} (v${st.profile_version || 1})</h3>
                <small class="text-muted">${escapeHtml(st.education || st.grade || '')} • Self-Level: <strong>${escapeHtml(st.self_reported_level || 'Not sure')}</strong></small>
              </div>
              <span class="tag tag-accent">${st.total_sessions} Sessions</span>
            </div>
            <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:6px;"><strong>Learning Goals:</strong> ${escapeHtml(st.learning_goals || 'Improve communication')}</p>
            <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:12px;"><strong>Topics:</strong> ${escapeHtml(st.learning_topics || st.interests || 'General Topics')}</p>
            
            <div style="margin-bottom:8px; font-size:0.8rem; color:#34d399;">
              <strong>Strengths:</strong> ${(st.strengths || []).join(", ") || "Active participant"}
            </div>
            <div style="margin-bottom:14px; font-size:0.8rem; color:#f43f5e;">
              <strong>Weaknesses:</strong> ${(st.weaknesses || []).join(", ") || "None recorded"}
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 12px; margin-bottom: 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div style="font-size:0.75rem; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px;">Initial Assessment Baseline</div>
                <span class="tag ${st.speaking_assessment_completed ? 'tag-success' : 'tag-accent'}" style="font-size:0.7rem;">
                  ${st.speaking_assessment_completed ? 'Verified Baseline' : 'Pending Speech Assessment'}
                </span>
              </div>
              <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.8rem;">
                <div><span class="text-muted">Grammar:</span> <strong>${st.baseline_grammar || 0}%</strong></div>
                <div><span class="text-muted">Vocab:</span> <strong>${st.baseline_vocabulary || 0}%</strong></div>
                <div><span class="text-muted">Fluency:</span> <strong>${st.baseline_fluency || 0}%</strong></div>
                <div><span class="text-muted">Pronunc:</span> <strong>${st.baseline_pronunciation || 0}%</strong></div>
                <div><span class="text-muted">Confidence:</span> <strong>${st.baseline_confidence || 0}%</strong></div>
                <div><span class="text-muted">Communic:</span> <strong>${st.baseline_communication || 0}%</strong></div>
              </div>
            </div>

            <div class="progress-list">
              <div class="progress-item">
                <div class="progress-labels"><span>Historical Fluency</span><strong>${st.fluency_score}%</strong></div>
                <div class="progress-bar"><div class="progress-fill fill-cyan" style="width:${st.fluency_score}%"></div></div>
              </div>
              <div class="progress-item">
                <div class="progress-labels"><span>Historical Grammar</span><strong>${st.grammar_score}%</strong></div>
                <div class="progress-bar"><div class="progress-fill fill-indigo" style="width:${st.grammar_score}%"></div></div>
              </div>
            </div>
          </div>
        `).join("");
      }
    }
  } catch (err) {
    console.error("Error loading students:", err);
  }
}

async function loadAssessments() {
  const container = document.getElementById("assessments-container");
  const countTag = document.getElementById("assessments-count-tag");
  if (!container) return;

  try {
    const res = await fetch(`/api/v1/students/${currentStudentId}/assessments`);
    if (!res.ok) return;
    const assessments = await res.json();
    if (countTag) countTag.textContent = `${assessments.length} Assessments`;

    if (assessments.length === 0) {
      container.innerHTML = `<div class="text-muted" style="padding:15px; text-align:center;">No speaking assessments recorded yet. Complete the 1-minute onboarding speech test to generate your baseline!</div>`;
      return;
    }

    container.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:14px;">
        ${assessments.map(a => `
          <div style="background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
              <div>
                <strong style="color:#818cf8; font-size:1.05rem;">Speaking Assessment #${a.id} — Level: ${escapeHtml(a.overall_level)}</strong>
                <div style="font-size:0.75rem; color:var(--text-muted);">${new Date(a.created_at).toLocaleString()} • Duration: ${a.duration_seconds || 0}s</div>
              </div>
              <span class="badge badge-accent">${escapeHtml(a.overall_level)}</span>
            </div>
            
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 12px; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 8px;">
              <div><small class="text-muted">Grammar:</small> <strong>${a.grammar_score}%</strong></div>
              <div><small class="text-muted">Vocabulary:</small> <strong>${a.vocabulary_score}%</strong></div>
              <div><small class="text-muted">Fluency:</small> <strong>${a.fluency_score}%</strong></div>
              <div><small class="text-muted">Pronunciation:</small> <strong>${a.pronunciation_score}%</strong></div>
              <div><small class="text-muted">Confidence:</small> <strong>${a.confidence_score}%</strong></div>
              <div><small class="text-muted">Communication:</small> <strong>${a.communication_score}%</strong></div>
            </div>

            <div style="font-size:0.85rem; line-height:1.45; margin-bottom:8px;">
              <div style="margin-bottom:4px;"><strong>Grammar Feedback:</strong> <span class="text-muted">${escapeHtml(a.grammar_feedback || 'Accurate structure')}</span></div>
              <div style="margin-bottom:4px;"><strong>Vocabulary Feedback:</strong> <span class="text-muted">${escapeHtml(a.vocabulary_feedback || 'Appropriate vocabulary')}</span></div>
              <div style="margin-bottom:4px;"><strong>Pronunciation:</strong> <span class="text-muted">${escapeHtml(a.pronunciation_feedback || 'Clear delivery')}</span></div>
              <div style="margin-bottom:4px;"><strong>Confidence / Delivery:</strong> <span class="text-muted">${escapeHtml(a.confidence_feedback || 'Observable delivery flow')}</span></div>
            </div>
            
            <div style="display:flex; gap:16px; font-size:0.8rem; flex-wrap:wrap; margin-top:8px;">
              <span style="color:#34d399;"><strong>Strengths:</strong> ${(a.strengths || []).join(", ") || "Active participation"}</span>
              <span style="color:#f43f5e;"><strong>Challenges:</strong> ${(a.weaknesses || []).join(", ") || "None"}</span>
            </div>
          </div>
        `).join("")}
      </div>
    `;
  } catch (err) {
    console.error("Error loading assessments:", err);
  }
}

async function loadStudentMistakes() {
  const container = document.getElementById("mistakes-container");
  if (!container) return;

  try {
    const res = await fetch(`/api/v1/students/${currentStudentId}/mistakes?limit=6`);
    if (!res.ok) return;
    const mistakes = await res.json();

    if (mistakes.length === 0) {
      container.innerHTML = `<div class="text-muted" style="padding:15px; text-align:center;">No recurring grammar flaws detected yet. Great job!</div>`;
      return;
    }

    container.innerHTML = `
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px;">
        ${mistakes.map(m => `
          <div style="background: rgba(244, 63, 94, 0.05); border: 1px solid rgba(244, 63, 94, 0.2); border-radius: 8px; padding: 12px;">
            <div style="font-size:0.85rem; font-weight:600; color:#fbbf24; margin-bottom:4px;">${escapeHtml(m.mistake_type).toUpperCase()}: ${escapeHtml(m.explanation || '')}</div>
            <div style="font-size:0.85rem; margin-bottom:6px;">
              <span style="color:#f43f5e; text-decoration:line-through;">${escapeHtml(m.error_text)}</span> ➔ <strong style="color:#34d399;">${escapeHtml(m.correction)}</strong>
            </div>
            <small class="text-muted">Utterance: "${escapeHtml(m.utterance)}"</small>
          </div>
        `).join("")}
      </div>
    `;
  } catch (err) {
    console.error("Error loading mistakes:", err);
  }
}

async function loadHistoricalProgress() {
  try {
    const res = await fetch(`/api/v1/students/${currentStudentId}/historical-progress`);
    if (!res.ok) return;
    const data = await res.json();
    const subTitle = document.getElementById("historical-progress-subtitle");

    if (!data.has_data || data.total_assessments === 0) {
      if (subTitle) subTitle.textContent = "No completed speaking assessments recorded yet";
      updateBar("fluency", null, "Source: No historical assessment");
      updateBar("grammar", null, "Source: No historical assessment");
      updateBar("vocab", null, "Source: No historical assessment");
      updateBar("conf", null, "Source: No historical assessment");
    } else {
      const lastSrc = data.sources && data.sources.length > 0 ? data.sources[data.sources.length - 1] : null;
      const srcText = lastSrc
        ? `Source: Assessment #${lastSrc.assessment_id} (${(lastSrc.session_id || '').substring(0, 10)}...)`
        : `Source: ${data.total_assessments} assessment(s)`;
      if (subTitle) subTitle.textContent = `Calculated from ${data.total_assessments} assessment(s) • Level: ${data.overall_level || 'Intermediate'}`;
      updateBar("fluency", data.fluency_score, srcText);
      updateBar("grammar", data.grammar_score, srcText);
      updateBar("vocab", data.vocabulary_score, srcText);
      updateBar("conf", data.confidence_score, srcText);
    }
  } catch (err) {
    console.error("Error loading historical progress:", err);
  }
}

function updateBar(key, val, sourceText = "") {
  const textEl = document.getElementById(`snap-${key}`);
  const barEl = document.getElementById(`bar-${key}`);
  const srcEl = document.getElementById(`snap-source-${key}`);

  if (val === null || val === undefined || isNaN(val)) {
    if (textEl) textEl.textContent = "Not assessed yet";
    if (barEl) barEl.style.width = "0%";
  } else {
    const rounded = Math.round(val);
    if (textEl) textEl.textContent = `${rounded}.0%`;
    if (barEl) barEl.style.width = `${rounded}%`;
  }

  if (srcEl && sourceText) {
    srcEl.textContent = sourceText;
  }
}

// -------------------------------------------------------------
// Conversation Stream & History
// -------------------------------------------------------------
async function loadConversations(searchQuery = null) {
  try {
    let url = `/api/v1/conversations?student_id=${currentStudentId}&limit=50`;
    if (searchQuery && searchQuery.trim()) {
      url += `&search=${encodeURIComponent(searchQuery.trim())}`;
    }

    const res = await fetch(url);
    if (!res.ok) return;
    const logs = await res.json();

    // Canonical chronological ordering: oldest to newest
    logs.sort((a, b) => (a.id || 0) - (b.id || 0));

    // 1. Render in Live Chat Box on Onboarding Studio
    const chatBox = document.getElementById("chat-messages-container");
    if (chatBox) {
      if (logs.length === 0) {
        chatBox.innerHTML = `
          <div class="text-muted" style="text-align:center; padding:40px;" id="chat-idle-placeholder">
            💤 Mizo is in snooze mode.<br>
            <small style="color:var(--text-dim); margin-top:8px; display:inline-block;">Say <strong>"Hey Mizo"</strong> to wake up and start onboarding.</small>
          </div>
        `;
      } else {
        chatBox.innerHTML = logs.map(msg => renderChatMessage(msg)).join("");
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    // 2. Render in Full Conversation History Tab
    const historyBox = document.getElementById("history-stream-box");
    if (historyBox) {
      if (logs.length === 0) {
        historyBox.innerHTML = `<div class="text-muted" style="text-align:center; padding:30px;">No conversation messages found.</div>`;
      } else {
        historyBox.innerHTML = logs.map(msg => renderHistoryItem(msg)).join("");
      }
    }
  } catch (err) {
    console.error("Error loading conversations:", err);
  }
}

function renderChatMessage(msg) {
  const isUser = msg.role === "user";
  const audioBtn = msg.audio_path
    ? `<button class="btn btn-secondary btn-sm" onclick="playAudio('${msg.audio_path}')" style="margin-top:6px; padding:4px 10px; font-size:0.75rem;">▶ Listen Speech</button>`
    : "";
  const corrections = (msg.grammar_errors && msg.grammar_errors.length > 0)
    ? `<div style="margin-top:6px; font-size:0.8rem; color:#fbbf24; background:rgba(0,0,0,0.2); padding:6px 10px; border-radius:6px;">
         <strong>Corrections:</strong> ${msg.grammar_errors.map(e => `${escapeHtml(e.error)} ➔ ${escapeHtml(e.correction)}`).join(", ")}
       </div>`
    : "";

  return `
    <div class="chat-bubble ${isUser ? 'user' : 'assistant'}">
      <div class="chat-avatar">${isUser ? '👤' : '🤖'}</div>
      <div class="chat-body">
        <div class="chat-meta">
          <strong>${isUser ? 'You' : 'Mizo'}</strong>
          <span>${new Date(msg.timestamp).toLocaleTimeString()}</span>
          ${!isUser && msg.provider_used ? `<span class="badge badge-accent" style="font-size:0.65rem;">${escapeHtml(msg.provider_used)}</span>` : ''}
        </div>
        <div class="chat-text">
          ${escapeHtml(msg.content)}
          ${corrections}
          ${audioBtn}
        </div>
      </div>
    </div>
  `;
}

function renderHistoryItem(msg) {
  const isUser = msg.role === "user";
  const audioBtn = msg.audio_path
    ? `<button class="btn btn-secondary btn-sm" onclick="playAudio('${msg.audio_path}')" style="margin-top:8px;">▶ Listen Audio</button>`
    : "";
  const corrections = (msg.grammar_errors && msg.grammar_errors.length > 0)
    ? `<div style="margin-top:8px; font-size:0.8rem; color:#fbbf24;">
         <strong>Corrections:</strong> ${msg.grammar_errors.map(e => `${escapeHtml(e.error)} ➔ ${escapeHtml(e.correction)}`).join(", ")}
       </div>`
    : "";

  const providerBadge = !isUser && msg.provider_used
    ? `<span class="badge ${msg.is_fallback ? 'badge-danger' : 'badge-accent'}" style="font-size:0.7rem; margin-left:8px;">
         ${escapeHtml(msg.provider_used)} ${msg.is_fallback ? '[FALLBACK]' : ''} ${msg.latency_ms ? `(${msg.latency_ms}ms)` : ''}
       </span>`
    : "";

  return `
    <div class="history-item ${isUser ? 'history-user' : 'history-assistant'}">
      <div class="history-header">
        <div>
          <strong>${isUser ? '👤 Learner' : '🤖 Mizo AI'}</strong>
          ${providerBadge}
        </div>
        <span>${new Date(msg.timestamp).toLocaleString()} (Session: ${escapeHtml(msg.session_id ? msg.session_id.substring(0, 8) : 'N/A')})</span>
      </div>
      <div class="history-content">${escapeHtml(msg.content)}</div>
      ${corrections}
      ${audioBtn}
    </div>
  `;
}

function playAudio(url) {
  const audio = new Audio(url);
  audio.play().catch(e => console.warn("Audio playback error:", e));
}

async function loadKnowledgeDocs() {
  try {
    const res = await fetch("/api/v1/knowledge/documents");
    if (!res.ok) return;
    const docs = await res.json();
    const container = document.getElementById("doc-list-container");
    const countTag = document.getElementById("doc-count-tag");
    if (countTag) countTag.textContent = `${docs.length} Documents`;
    if (!container) return;

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
    if (!tbody) return;

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
// -------------------------------------------------------------
// Voice State Machine & Two-Way Continuous Engine
// -------------------------------------------------------------
function setSilenceBadge(statusText, type = "normal") {
  const badge = document.getElementById("silence-status-badge");
  if (!badge) return;
  badge.textContent = statusText;
  badge.className = `silence-badge ${type}`;
}

function setVoiceState(newState, reason = "") {
  console.log(`[Voice SM] Transition: ${currentVoiceState} ➔ ${newState} (${reason})`);
  currentVoiceState = newState;

  if (newState === VoiceState.WAKE_LISTENING || newState === VoiceState.STANDBY || newState === VoiceState.WAKE_ONLY || newState === VoiceState.SNOOZED) {
    console.log("[VOICE] State: WAKE_LISTENING");
    console.log("[VOICE] Wake listener: ACTIVE");
    console.log("[VOICE] Microphone: ACTIVE");
  } else if (newState === VoiceState.ACTIVE_CONVERSATION || newState === VoiceState.ACTIVE_LISTENING || newState === VoiceState.READY || newState === VoiceState.CONVERSATION || newState === VoiceState.ONBOARDING) {
    console.log("[VOICE] State: ACTIVE_CONVERSATION");
    console.log("[VOICE] State: LISTENING");
    console.log("[VOICE] Microphone: ACTIVE");
  }

  const stateBadge = document.getElementById("voice-state-badge");
  if (stateBadge) {
    if (isMicMuted) {
      stateBadge.className = "voice-state-badge muted";
      stateBadge.textContent = "MIC MUTED";
    } else {
      const stateMap = {
        [VoiceState.WAKE_LISTENING]: { cls: "wake-only", label: "WAKE_LISTENING • Say 'Hey Mizo'" },
        [VoiceState.STANDBY]: { cls: "wake-only", label: "WAKE_LISTENING • Say 'Hey Mizo'" },
        [VoiceState.SNOOZED]: { cls: "wake-only", label: "WAKE_LISTENING • Say 'Hey Mizo'" },
        [VoiceState.WAKE_ONLY]: { cls: "wake-only", label: "WAKE_LISTENING • Say 'Hey Mizo'" },
        [VoiceState.WAKE_DETECTED]: { cls: "active-listening", label: "WAKE_DETECTED" },
        [VoiceState.ACTIVE_CONVERSATION]: { cls: "active-listening", label: "ACTIVE_CONVERSATION" },
        [VoiceState.ONBOARDING]: { cls: "active-listening", label: "ONBOARDING" },
        [VoiceState.READY]: { cls: "active-listening", label: "READY" },
        [VoiceState.CONVERSATION]: { cls: "active-listening", label: "READY" },
        [VoiceState.ACTIVE_LISTENING]: { cls: "active-listening", label: "READY" },
        [VoiceState.CAPTURING]: { cls: "capturing", label: "CAPTURING" },
        [VoiceState.PROCESSING]: { cls: "processing", label: "PROCESSING" },
        [VoiceState.SPEAKING]: { cls: "speaking", label: "SPEAKING" },
        [VoiceState.STOPPING]: { cls: "stopping", label: "STOPPING" },
        [VoiceState.ERROR_RECOVERY]: { cls: "processing", label: "RECOVERING" },
        [VoiceState.ASSESSMENT_RECORDING]: { cls: "assessment-recording", label: "ASSESSMENT_RECORDING" }
      };
      const info = stateMap[newState] || { cls: "wake-only", label: newState };
      stateBadge.className = `voice-state-badge ${info.cls}`;
      stateBadge.textContent = info.label;
    }
  }

  // State-specific lifecycle management
  if (newState === VoiceState.STANDBY || newState === VoiceState.WAKE_ONLY || newState === VoiceState.SNOOZED || newState === VoiceState.STOPPING) {
    clearTimeout(inactivityTimerId);
    silenceStartTimestamp = null;
    speechDetected = false;

    // Halt audio playback immediately
    const player = document.getElementById("sim-audio-player");
    if (player) {
      player.pause();
      player.currentTime = 0;
    }
    const waveform = document.getElementById("sim-waveform");
    if (waveform) waveform.classList.remove("playing");
    isMizoSpeaking = false;
    isMikazaSpeaking = false;

    // Abort in-flight network requests
    if (currentAbortController) {
      try { currentAbortController.abort(); } catch(e) {}
      currentAbortController = null;
    }

    // Stop recording if active
    if (isRecording && mediaRecorder && mediaRecorder.state !== "inactive") {
      try { mediaRecorder.stop(); } catch(e) {}
      isRecording = false;
    }

    // Stop assessment HUD & timer if open
    hideAssessmentHUD();

    setSilenceBadge(isMicMuted ? "Microphone Muted" : "💤 SNOOZED • Say 'Mizo'", "normal");

    // Keep local wake detector active in standby
    if (isWakeWordEnabled && !isMicMuted) {
      initWakeWordListener();
    }
  } else if (newState === VoiceState.ACTIVE_LISTENING || newState === VoiceState.READY || newState === VoiceState.CONVERSATION || newState === VoiceState.ONBOARDING) {
    // Stop local wake detector so it doesn't conflict with conversational audio
    if (wakeWordRecognizer) {
      try { wakeWordRecognizer.abort(); } catch(e) {}
    }
    hideAssessmentHUD();
    setSilenceBadge("🟢 Ready (Listening...)", "playing");
    startInactivityCountdown();
  } else if (newState === VoiceState.CAPTURING) {
    clearTimeout(inactivityTimerId);
    setSilenceBadge("🗣️ Capturing speech...", "speaking");
  } else if (newState === VoiceState.PROCESSING) {
    clearTimeout(inactivityTimerId);
    setSilenceBadge("⚡ Mizo thinking & answering...", "thinking");
  } else if (newState === VoiceState.SPEAKING) {
    clearTimeout(inactivityTimerId);
    setSilenceBadge("🔊 Mizo speaking...", "playing");
  } else if (newState === VoiceState.ERROR_RECOVERY) {
    clearTimeout(inactivityTimerId);
    setSilenceBadge("⚠️ Recovering connection...", "normal");
  } else if (newState === VoiceState.ASSESSMENT_RECORDING) {
    if (wakeWordRecognizer) {
      try { wakeWordRecognizer.abort(); } catch(e) {}
    }
    clearTimeout(inactivityTimerId);
    silenceStartTimestamp = null;
    assessmentSpeechDetected = false;
    assessmentSilenceStart = null;
    setSilenceBadge("🎙️ Recording Assessment (~1 Min)", "speaking");
    showAssessmentHUD();
  }
}

function startInactivityCountdown() {
  clearTimeout(inactivityTimerId);
  const timeoutMs = inactivityTimeoutSeconds * 1000;
  inactivityTimerId = setTimeout(() => {
    if (currentVoiceState === VoiceState.ACTIVE_LISTENING) {
      console.log(`[Voice SM] 10s Inactivity reached with no user speech. Entering SNOOZE mode. Audio monitoring remains active.`);
      showToast(`💤 10s Inactivity. Entered Sleep. Say 'Mizo' to wake up.`);
      // Idle time alone must not make an active conversation reject the next turn.
      // Explicit stop commands remain the only transition into standby.
      setSilenceBadge("Ready (Listening...)", "playing");
      startInactivityCountdown();
    }
  }, timeoutMs);
}

function detectMizoIntent(text) {
  if (!text || !text.trim()) {
    console.log(`[WAKE] Raw transcription: "${text}"\n[WAKE] Normalized: ""\n[WAKE] Similarity to "mizo": 0.00\n[WAKE] Intent: UNKNOWN`);
    return "UNKNOWN";
  }
  const norm = text.toLowerCase().replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
  const words = norm.split(" ").filter(Boolean);
  if (words.length === 0) {
    console.log(`[WAKE] Raw transcription: "${text}"\n[WAKE] Normalized: ""\n[WAKE] Similarity to "mizo": 0.00\n[WAKE] Intent: UNKNOWN`);
    return "UNKNOWN";
  }

  // 1. Check stop commands
  const stopPhrases = [
    "stop mizo", "mizo stop", "stop meeso", "meeso stop", "stop miso", "miso stop",
    "stop mikaza", "mikaza stop", "stop megaza", "megaza stop", "stop", "stop listening",
    "stop talking", "stop now", "stop please", "please stop"
  ];
  if (stopPhrases.includes(norm) || stopPhrases.some(p => norm.startsWith(p + " "))) {
    console.log(`[WAKE] Raw transcription: "${text}"\n[WAKE] Normalized: "${norm}"\n[WAKE] Similarity to "mizo": 0.00\n[WAKE] Intent: STOP`);
    return "STOP";
  }

  // 2. Helper to test Mizo word phonetically and fuzzily
  function isMizoWord(w) {
    if (!w) return false;
    if (["mizo", "miso", "meeso", "meso", "mezo", "mizzo", "meiso", "mizu", "myzo"].includes(w)) return true;
    if (["mikaza", "megaza", "mikasa", "mikkaza", "makaza", "micaza"].includes(w)) return true;
    let p = w.replace(/ee|ei|ey|ea|y/g, "i");
    if (p.startsWith("me") && (p.length === 4 || p.length === 5)) p = "mi" + p.slice(2);
    p = p.replace(/zz|ss|s/g, "z");
    if (p.endsWith("u")) p = p.slice(0, -1) + "o";
    p = p.replace(/(.)\1+/g, "$1");
    if (p === "mizo") return true;
    return false;
  }

  let isWake = false;
  let bestSim = 0.0;

  for (const w of words) {
    if (isMizoWord(w)) {
      bestSim = 1.0;
      break;
    }
  }

  // Condition A: Single word (e.g. "Mizo", "Miso", "Meeso")
  if (words.length === 1 && isMizoWord(words[0])) {
    isWake = true;
    bestSim = 1.0;
  }
  // Condition B: Natural greeting followed by Mizo (e.g. "Hey Mizo", "Hello Mizo", "Hi Mizo")
  else if (["hey", "hi", "hello", "yo", "ok", "okay"].includes(words[0]) && words.length >= 2 && isMizoWord(words[1])) {
    isWake = true;
    bestSim = 1.0;
  }
  // Condition C: Utterance starting with Mizo (e.g. "Mizo, are you there?", "Mizo, wake up")
  else if (isMizoWord(words[0])) {
    isWake = true;
    bestSim = 1.0;
  }
  // Condition D: Addressing phrases ending with Mizo (e.g. "Wake up, Mizo", "Can you hear me, Mizo", "What's up, Mizo")
  else if (isMizoWord(words[words.length - 1])) {
    const addressingStarters = ["wake up", "are you there", "can you hear me", "whats up", "what is up", "hello", "hey", "hi", "talk to me", "listen to me"];
    if (addressingStarters.some(s => norm.includes(s))) {
      isWake = true;
      bestSim = 1.0;
    }
  }
  // Condition E: Phrase containing "wake up" + Mizo
  else if (norm.includes("wake up") && words.some(w => isMizoWord(w))) {
    isWake = true;
    bestSim = 0.95;
  }

  const intent = isWake ? "WAKE" : "NORMAL";
  console.log(`[WAKE] Raw transcription: "${text}"\n[WAKE] Normalized: "${norm}"\n[WAKE] Similarity to "mizo": ${bestSim.toFixed(2)}\n[WAKE] Intent: ${intent}`);
  return intent;
}

function extractMizoTrailingText(fullTranscript) {
  const norm = fullTranscript.toLowerCase().replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
  const words = norm.split(" ").filter(Boolean);
  const rawWords = fullTranscript.trim().split(/\s+/).filter(Boolean);

  function isMizoWord(w) {
    if (!w) return false;
    const clean = w.toLowerCase().replace(/[^\w]/g, "");
    if (["mizo", "miso", "meeso", "meso", "mezo", "mizzo", "meiso", "mizu", "myzo"].includes(clean)) return true;
    if (["mikaza", "megaza", "mikasa", "mikkaza", "makaza", "micaza"].includes(clean)) return true;
    let p = clean.replace(/ee|ei|ey|ea|y/g, "i");
    if (p.startsWith("me") && (p.length === 4 || p.length === 5)) p = "mi" + p.slice(2);
    p = p.replace(/zz|ss|s/g, "z");
    if (p.endsWith("u")) p = p.slice(0, -1) + "o";
    p = p.replace(/(.)\1+/g, "$1");
    return p === "mizo";
  }

  let trailing = "";
  if (words.length >= 2 && ["hey", "hi", "hello", "yo", "ok", "okay"].includes(words[0]) && isMizoWord(words[1])) {
    trailing = rawWords.slice(2).join(" ").trim();
  } else if (words.length >= 1 && isMizoWord(words[0])) {
    trailing = rawWords.slice(1).join(" ").trim();
  } else if (words.length >= 1 && isMizoWord(words[words.length - 1])) {
    trailing = "";
  }

  const cleanTrailing = trailing.toLowerCase().replace(/[^\w\s]/g, "").trim();
  if (["are you there", "wake up", "can you hear me", "whats up", "what is up", "please wake up"].includes(cleanTrailing)) {
    trailing = "";
  }
  return trailing;
}

function isVoiceIdleState(state) {
  return state === VoiceState.WAKE_LISTENING || state === VoiceState.STANDBY || state === VoiceState.WAKE_ONLY || state === VoiceState.SNOOZED;
}

function handleWakeWordDetected(fullTranscript) {
  if (isMicMuted || !isVoiceIdleState(currentVoiceState)) return;

  console.log(`[VOICE] Heard: "${fullTranscript}"`);
  console.log(`[VOICE] Wake word detected: Mizo`);
  console.log(`[VOICE] Wake word detected`);
  console.log(`[VOICE] Waking assistant`);
  console.log(`[VOICE] State: ACTIVE_CONVERSATION`);

  const remainingText = extractMizoTrailingText(fullTranscript);

  showToast("🎙️ 'Mizo' recognized! Active mode enabled.");

  if (remainingText.length > 0) {
    // Treat trailing words as direct user input
    setVoiceState(VoiceState.ACTIVE_CONVERSATION, "wake_with_speech");
    const textInput = document.getElementById("sim-text-input");
    if (textInput) textInput.value = remainingText;
    sendTextMessage(remainingText);
  } else {
    // Wake phrase alone -> transition to ACTIVE_CONVERSATION & acknowledge
    setVoiceState(VoiceState.ACTIVE_CONVERSATION, "wake_phrase");
    console.log("[VOICE] Listening for user request");
    const chatBox = document.getElementById("chat-messages-container");
    const hasHistory = chatBox && chatBox.querySelectorAll(".chat-bubble").length > 1;
    if (!hasHistory) {
      triggerMizoStartGreeting();
    } else {
      sendTextMessage("Hey Mizo");
    }
  }
}

function executeMizoStop() {
  console.log("[VOICE] Stop command detected");
  console.log("[Voice SM] 'Mizo stop' control command received.");
  setVoiceState(VoiceState.STOPPING, "mizo_stop_command");

  // 1. Immediately halt audio playback
  const player = document.getElementById("sim-audio-player");
  if (player) {
    player.pause();
    player.currentTime = 0;
  }
  const waveform = document.getElementById("sim-waveform");
  if (waveform) waveform.classList.remove("playing");
  isMizoSpeaking = false;
  isMikazaSpeaking = false;

  // 2. Abort any running network/generation requests
  if (currentAbortController) {
    try { currentAbortController.abort(); } catch(e) {}
    currentAbortController = null;
  }

  // 3. Stop recording immediately
  if (isRecording && mediaRecorder && mediaRecorder.state !== "inactive") {
    try { mediaRecorder.stop(); } catch(e) {}
  }
  isRecording = false;

  // 4. Return to WAKE_LISTENING mode with microphone & listener active
  setVoiceState(VoiceState.WAKE_LISTENING, "mizo_stop_command");
  showToast("🛑 'Mizo stop' executed. Wake listening active.");
}
const executeMikazaStop = executeMizoStop;

function toggleMicrophoneMute() {
  isMicMuted = !isMicMuted;
  const btn = document.getElementById("btn-toggle-mic-mute");
  if (isMicMuted) {
    if (btn) {
      btn.textContent = "🔇 Mic Muted";
      btn.style.color = "#f43f5e";
    }
    if (isRecording) stopVoiceRecording();
    if (micMediaStream) {
      micMediaStream.getTracks().forEach(t => t.stop());
    }
    if (wakeWordRecognizer) {
      try { wakeWordRecognizer.abort(); } catch(e) {}
    }
    setVoiceState(VoiceState.WAKE_ONLY, "mic_muted");
    showToast("Microphone Muted / Disabled");
  } else {
    if (btn) {
      btn.textContent = "🎙️ Mic Active";
      btn.style.color = "";
    }
    setVoiceState(VoiceState.WAKE_ONLY, "mic_unmuted");
    showToast("Microphone Active — Standby listening for 'Mizo'");
  }
}

async function startTwoWayVoiceMode() {
  isTwoWayVoiceMode = true;
  const btn = document.getElementById("btn-toggle-voice-mode");
  const label = document.getElementById("voice-mode-label");
  if (btn) btn.classList.add("active");
  if (label) label.textContent = "Live Conversation Active";

  setVoiceState(VoiceState.ACTIVE_LISTENING, "user_toggle_voice_mode");
  showToast("🎙️ Live Voice Mode Active! Mizo is speaking & listening.");

  const chatBox = document.getElementById("chat-messages-container");
  const hasHistory = chatBox && chatBox.querySelectorAll(".chat-bubble").length > 1;

  if (!hasHistory && !isMizoSpeaking) {
    await triggerMizoStartGreeting();
  } else {
    await startVoiceRecording(true);
  }
}

function stopTwoWayVoiceMode() {
  isTwoWayVoiceMode = false;
  const btn = document.getElementById("btn-toggle-voice-mode");
  const label = document.getElementById("voice-mode-label");
  if (btn) btn.classList.remove("active");
  if (label) label.textContent = "Start Live Voice Conversation";

  setVoiceState(VoiceState.WAKE_ONLY, "user_paused_voice_mode");
  const volBar = document.getElementById("volume-meter-bar");
  if (volBar) volBar.style.width = "0%";

  showToast("Live Voice Mode Paused (Standby)");
}

async function triggerMizoStartGreeting() {
  setVoiceState(VoiceState.PROCESSING, "fetching_greeting");
  try {
    currentAbortController = new AbortController();
    const res = await fetch("/api/v1/esp32/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: currentStudentId,
        action: "start"
      }),
      signal: currentAbortController.signal
    });
    const data = await res.json();
    currentAbortController = null;
    handleInteractionResponse(data);
  } catch (err) {
    if (err.name === "AbortError") return;
    currentAbortController = null;
    console.error("Failed to trigger start greeting:", err);
    setVoiceState(VoiceState.ACTIVE_LISTENING, "greeting_failed");
    startVoiceRecording(true);
  }
}
const triggerMikazaStartGreeting = triggerMizoStartGreeting;

function initAudioContextVAD(stream) {
  try {
    if (!audioContext || audioContext.state === "closed") {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 512;
    analyserNode.smoothingTimeConstant = 0.25;

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyserNode);

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    speechDetected = false;
    silenceStartTimestamp = null;
    calibrationFrames = 0;
    ambientNoiseFloor = 0.005;

    const checkVolume = () => {
      if (isMicMuted || currentVoiceState === VoiceState.WAKE_ONLY) {
        vadAnimationId = requestAnimationFrame(checkVolume);
        return;
      }

      analyserNode.getByteTimeDomainData(dataArray);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        const val = (dataArray[i] - 128) / 128;
        sum += val * val;
      }
      const rms = Math.sqrt(sum / bufferLength);

      if (calibrationFrames < 25) {
        ambientNoiseFloor = (ambientNoiseFloor * calibrationFrames + rms) / (calibrationFrames + 1);
        calibrationFrames++;
      }

      const currentSpeechThreshold = Math.max(0.009, ambientNoiseFloor * 1.55);

      const volPercent = Math.min(100, Math.round(rms * 450));
      const volBar = document.getElementById("volume-meter-bar");
      if (volBar) volBar.style.width = `${volPercent}%`;

      const now = Date.now();
      const elapsedSinceStart = recordStartTime ? (now - recordStartTime) / 1000 : 0;

      // 1. Barge-In / Interruption while Mizo is speaking
      if (currentVoiceState === VoiceState.SPEAKING || isMizoSpeaking || isMikazaSpeaking) {
        if (rms > currentSpeechThreshold * 1.5) {
          console.log("[Voice SM] Barge-in detected: User speaking while Mizo is speaking. Halting TTS immediately.");
          const player = document.getElementById("sim-audio-player");
          if (player) {
            player.pause();
            player.currentTime = 0;
          }
          const waveform = document.getElementById("sim-waveform");
          if (waveform) waveform.classList.remove("playing");
          isMizoSpeaking = false;
          isMikazaSpeaking = false;
          setVoiceState(VoiceState.CAPTURING, "user_interruption_barge_in");
          startVoiceRecording(true);
        }
        vadAnimationId = requestAnimationFrame(checkVolume);
        return;
      }

      // 2. Dedicated Assessment Recording: speech detected then 3s continuous silence auto-submits
      if (currentVoiceState === VoiceState.ASSESSMENT_RECORDING) {
        if (rms > currentSpeechThreshold) {
          assessmentSpeechDetected = true;
          assessmentSilenceStart = null;
          setSilenceBadge("🎙️ Recording Assessment (Speech detected)...", "speaking");
        } else if (assessmentSpeechDetected) {
          if (!assessmentSilenceStart) {
            assessmentSilenceStart = now;
          }
          const silenceDuration = (now - assessmentSilenceStart) / 1000;
          const remaining = Math.max(0, (assessmentSilenceTimeoutSeconds - silenceDuration)).toFixed(1);
          setSilenceBadge(`⏳ Assessment silence detected (auto-submit in ${remaining}s)`, "silence-countdown");

          if (silenceDuration >= assessmentSilenceTimeoutSeconds) {
            const elapsedSecs = assessmentStartTime ? Math.round((Date.now() - assessmentStartTime) / 1000) : 0;
            if (elapsedSecs >= 15) {
              console.log(`[Assessment VAD] Continuous silence threshold (${assessmentSilenceTimeoutSeconds}s) reached after >=15s speech. Auto-submitting.`);
              finishSpeakingAssessment(true);
              return;
            } else {
              assessmentSilenceStart = null;
              setSilenceBadge("🎙️ Take your time. Please speak freely for about a minute on the topic...", "normal");
            }
          }
        } else {
          setSilenceBadge("🎙️ Recording Assessment (Speak continuously)...", "normal");
        }
        vadAnimationId = requestAnimationFrame(checkVolume);
        return;
      }

      // 3. Normal Active Turn VAD (Turn ending silence timer)
      if (currentVoiceState === VoiceState.ACTIVE_LISTENING || currentVoiceState === VoiceState.CAPTURING) {
        if (rms > currentSpeechThreshold) {
          if (currentVoiceState === VoiceState.ACTIVE_LISTENING) {
            setVoiceState(VoiceState.CAPTURING, "speech_detected");
          }
          speechDetected = true;
          silenceStartTimestamp = null;
          setSilenceBadge("🗣️ Capturing speech...", "speaking");
        } else if (speechDetected || elapsedSinceStart > 1.5) {
          if (!silenceStartTimestamp) {
            silenceStartTimestamp = now;
          }

          const silenceDuration = (now - silenceStartTimestamp) / 1000;
          const remaining = Math.max(0, (turnEndingSilenceSeconds - silenceDuration)).toFixed(1);
          setSilenceBadge(`⏳ Silence detected (auto-send in ${remaining}s)`, "silence-countdown");

          if (silenceDuration >= turnEndingSilenceSeconds) {
            console.log(`[VAD] Turn ending silence threshold (${turnEndingSilenceSeconds}s) reached.`);
            setVoiceState(VoiceState.PROCESSING, "turn_silence_threshold");
            stopVoiceRecording();
            return;
          }
        }
      }

      vadAnimationId = requestAnimationFrame(checkVolume);
    };

    vadAnimationId = requestAnimationFrame(checkVolume);
  } catch (err) {
    console.warn("VAD Web Audio initialization error:", err);
  }
}

async function startVoiceRecording(isAuto = false) {
  if (isRecording || isMicMuted || isMizoSpeaking || isMikazaSpeaking || isProcessingBackend) return;

  const btnRecord = document.getElementById("btn-mic-record");
  const recordBtnText = document.getElementById("record-btn-text");
  const recordStatus = document.getElementById("record-status-text");
  const timerEl = document.getElementById("record-hud-timer");

  try {
    micMediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    let mimeType = "audio/webm;codecs=opus";
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = "audio/webm";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "";
      }
    }

    mediaRecorder = mimeType ? new MediaRecorder(micMediaStream, { mimeType }) : new MediaRecorder(micMediaStream);
    audioChunks = [];
    recordStartTime = Date.now();

    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      clearInterval(recordTimerInterval);
      if (vadAnimationId) cancelAnimationFrame(vadAnimationId);

      const durationSecs = Math.round((Date.now() - recordStartTime) / 1000);
      lastRecordedDuration = durationSecs;

      const recordedMime = mediaRecorder.mimeType || "audio/webm";
      const audioBlob = new Blob(audioChunks, { type: recordedMime });
      lastRecordedBlob = audioBlob;

      // Retain micMediaStream active so AudioContext VAD remains continuously available for barge-in
      // Tracks are only terminated if user explicitly mutes the microphone.

      // If user said "Mizo stop", do not send
      if (currentVoiceState === VoiceState.WAKE_ONLY) return;

      // Check if user spoke before sending
      if (!speechDetected && durationSecs < 1.8 && currentVoiceState !== VoiceState.ASSESSMENT_RECORDING) {
        setTimeout(() => {
          if (currentVoiceState === VoiceState.ACTIVE_LISTENING && !isMikazaSpeaking) startVoiceRecording(true);
        }, 500);
        return;
      }

      sendAudioToBackend(audioBlob, currentVoiceState === VoiceState.ASSESSMENT_RECORDING);
    };

    mediaRecorder.start();
    isRecording = true;

    if (currentVoiceState !== VoiceState.ASSESSMENT_RECORDING) {
      setVoiceState(VoiceState.ACTIVE_LISTENING, "recording_started");
    }

    if (btnRecord) btnRecord.classList.add("recording");
    if (recordBtnText) recordBtnText.textContent = "Listening";
    if (recordStatus) recordStatus.textContent = "🟢 Live microphone active • Speak naturally";
    if (timerEl) timerEl.classList.add("recording");

    initAudioContextVAD(micMediaStream);

    const formatTime = (secs) => {
      const m = Math.floor(secs / 60).toString().padStart(2, "0");
      const s = (secs % 60).toString().padStart(2, "0");
      return `${m}:${s}`;
    };

    recordTimerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
      if (timerEl) timerEl.textContent = formatTime(elapsed);
    }, 1000);

  } catch (err) {
    console.error("Microphone access error:", err);
    alert("Microphone permission denied or device unavailable.\n\nYou can still type your answers in the chat input!");
    if (recordStatus) recordStatus.textContent = "Microphone unavailable. Use chat input.";
    setVoiceState(VoiceState.WAKE_ONLY, "mic_denied");
  }
}

function stopVoiceRecording() {
  if (!isRecording || !mediaRecorder) return;

  const btnRecord = document.getElementById("btn-mic-record");
  const recordBtnText = document.getElementById("record-btn-text");
  const recordStatus = document.getElementById("record-status-text");
  const timerEl = document.getElementById("record-hud-timer");

  try {
    if (mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  } catch (e) {}

  isRecording = false;
  if (btnRecord) btnRecord.classList.remove("recording");
  if (recordBtnText) recordBtnText.textContent = "Voice";
  if (recordStatus) recordStatus.textContent = "⚡ Mizo processing your voice...";
  if (timerEl) timerEl.classList.remove("recording");
}

function showAssessmentHUD(topic = "") {
  const hud = document.getElementById("assessment-hud-box");
  const topicEl = document.getElementById("assessment-hud-topic");
  const timerEl = document.getElementById("assessment-hud-timer");
  const progressFill = document.getElementById("assessment-progress-fill");

  if (!hud) return;
  hud.style.display = "block";
  if (topic && topicEl) topicEl.textContent = `Topic: ${topic}`;

  isAssessmentFinalizing = false;
  assessmentStartTime = Date.now();
  assessmentSpeechDetected = false;
  assessmentSilenceStart = null;
  if (progressFill) progressFill.style.width = "0%";

  clearInterval(assessmentTimerInterval);
  assessmentTimerInterval = setInterval(() => {
    const elapsedSecs = Math.floor((Date.now() - assessmentStartTime) / 1000);
    const m = Math.floor(elapsedSecs / 60).toString().padStart(2, "0");
    const s = (elapsedSecs % 60).toString().padStart(2, "0");
    if (timerEl) timerEl.textContent = `${m}:${s} / ~01:00`;

    const pct = Math.min(100, Math.round((elapsedSecs / 60) * 100));
    if (progressFill) progressFill.style.width = `${pct}%`;

    // Maximum upper limit ~120s: auto-finalize sample
    if (elapsedSecs >= 120) {
      clearInterval(assessmentTimerInterval);
      showToast("Maximum assessment time reached (120s). Finalizing speech sample.");
      finishSpeakingAssessment(true);
    }
  }, 1000);
}

function hideAssessmentHUD() {
  const hud = document.getElementById("assessment-hud-box");
  if (hud) hud.style.display = "none";
  clearInterval(assessmentTimerInterval);
}

function finishSpeakingAssessment(isAuto = false) {
  if (isAssessmentFinalizing) {
    console.log("[Assessment VAD] Assessment already finalizing, ignoring duplicate submit invocation.");
    return;
  }
  const elapsedSecs = assessmentStartTime ? Math.round((Date.now() - assessmentStartTime) / 1000) : 0;
  hideAssessmentHUD();

  if (elapsedSecs < 10 && !isAuto) {
    showToast(`Speech sample was only ${elapsedSecs}s. Please speak for about 1 minute on the topic.`, true);
    const modal = document.getElementById("modal-short-recording");
    if (modal) modal.style.display = "flex";
    return;
  }

  isAssessmentFinalizing = true;
  stopVoiceRecording();
}

function setupAudioBench() {
  // 1. Toggle Live Continuous Voice Mode Button
  const btnVoiceMode = document.getElementById("btn-toggle-voice-mode");
  if (btnVoiceMode) {
    btnVoiceMode.addEventListener("click", () => {
      if (isTwoWayVoiceMode) {
        stopTwoWayVoiceMode();
      } else {
        startTwoWayVoiceMode();
      }
    });
  }

  // 2. Microphone Push-to-Talk / Toggle Record Button
  const btnMic = document.getElementById("btn-mic-record");
  if (btnMic) {
    btnMic.addEventListener("click", () => {
      if (isRecording) {
        stopVoiceRecording();
      } else {
        startVoiceRecording(false);
      }
    });
  }

  // 3. Send Text Message Button & Input (Enter Key)
  const btnSend = document.getElementById("btn-sim-send");
  const textInput = document.getElementById("sim-text-input");
  if (btnSend) {
    btnSend.addEventListener("click", () => sendTextMessage());
  }
  if (textInput) {
    textInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        sendTextMessage();
      }
    });
  }

  // 4. Toggle Microphone Mute / Disable
  const btnMute = document.getElementById("btn-toggle-mic-mute");
  if (btnMute) {
    btnMute.addEventListener("click", () => toggleMicrophoneMute());
  }

  // 5. "Mizo stop" control command button
  const btnStop = document.getElementById("btn-mizo-stop") || document.getElementById("btn-mikaza-stop");
  if (btnStop) {
    btnStop.addEventListener("click", () => executeMizoStop());
  }

  // 6. Development Fallback "Mizo" button
  const btnDevWake = document.getElementById("btn-dev-wake-phrase");
  if (btnDevWake) {
    btnDevWake.addEventListener("click", () => handleWakeWordDetected("Mizo"));
  }

  // 7. Wake Word Toggle Checkbox
  const chkWake = document.getElementById("chk-wake-word");
  if (chkWake) {
    chkWake.addEventListener("change", () => {
      isWakeWordEnabled = chkWake.checked;
      if (!isWakeWordEnabled && wakeWordRecognizer) {
        try { wakeWordRecognizer.abort(); } catch(e) {}
      } else if (isWakeWordEnabled && currentVoiceState === VoiceState.WAKE_ONLY && !isMicMuted) {
        initWakeWordListener();
      }
      showToast(isWakeWordEnabled ? "Wake word detection active ('Mizo')" : "Wake word detection paused");
    });
  }

  // 8. Turn Ending Silence Config Input (default 3.0s, range 1-10s)
  const settingTurnSilence = document.getElementById("setting-turn-silence");
  if (settingTurnSilence) {
    settingTurnSilence.value = turnEndingSilenceSeconds;
    settingTurnSilence.addEventListener("change", () => {
      const val = parseFloat(settingTurnSilence.value);
      if (!isNaN(val) && val >= 1.0 && val <= 10.0) {
        turnEndingSilenceSeconds = val;
        showToast(`Turn silence timer set to ${turnEndingSilenceSeconds}s`);
      } else {
        settingTurnSilence.value = turnEndingSilenceSeconds;
      }
    });
  }

  // 9. Inactivity Standby Timeout Config Input (default 12s, range 10-15s)
  const settingInactivity = document.getElementById("setting-inactivity-timeout");
  if (settingInactivity) {
    settingInactivity.value = inactivityTimeoutSeconds;
    settingInactivity.addEventListener("change", () => {
      const val = parseFloat(settingInactivity.value);
      if (!isNaN(val) && val >= 10.0 && val <= 15.0) {
        inactivityTimeoutSeconds = val;
        showToast(`Inactivity standby timeout set to ${inactivityTimeoutSeconds}s`);
      } else {
        settingInactivity.value = inactivityTimeoutSeconds;
      }
    });
  }

  // 10. Finish & Restart Assessment Sample Buttons
  const btnFinish = document.getElementById("btn-finish-assessment");
  if (btnFinish) {
    btnFinish.addEventListener("click", () => finishSpeakingAssessment());
  }
  const btnCancel = document.getElementById("btn-cancel-assessment");
  if (btnCancel) {
    btnCancel.addEventListener("click", () => {
      if (isRecording) stopVoiceRecording();
      setTimeout(() => {
        setVoiceState(VoiceState.ASSESSMENT_RECORDING, "restart_sample");
        startVoiceRecording(true);
      }, 400);
      showToast("Restarting speaking assessment sample...");
    });
  }
}

function initWakeWordListener() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    console.log("[Wake Word] Local browser SpeechRecognition not supported in this environment; development fallback button available.");
    return;
  }

  try {
    if (wakeWordRecognizer) {
      try { wakeWordRecognizer.abort(); } catch (e) {}
    }

    wakeWordRecognizer = new SpeechRec();
    wakeWordRecognizer.continuous = true;
    wakeWordRecognizer.interimResults = false;
    wakeWordRecognizer.lang = "en-US";

    wakeWordRecognizer.onresult = (event) => {
      if (!isWakeWordEnabled || isMicMuted || !isVoiceIdleState(currentVoiceState)) return;

      const lastResultIndex = event.results.length - 1;
      const transcript = event.results[lastResultIndex][0].transcript.trim();

      const intent = detectMizoIntent(transcript);
      if (intent === "STOP") {
        executeMikazaStop();
        return;
      }
      if (intent === "WAKE") {
        handleWakeWordDetected(transcript);
      } else {
        console.log(`[VOICE] Heard: "${transcript}"`);
        console.log("[VOICE] No wake word. Ignoring.");
      }
    };

    wakeWordRecognizer.onerror = (e) => {
      if (e.error !== "no-speech" && e.error !== "aborted") {
        console.log("[Wake Word Engine]", e.error);
      }
      if (isWakeWordEnabled && !isMicMuted && isVoiceIdleState(currentVoiceState)) {
        setTimeout(() => {
          try { wakeWordRecognizer.start(); } catch (err) {}
        }, 800);
      }
    };

    wakeWordRecognizer.onend = () => {
      if (isWakeWordEnabled && !isMicMuted && isVoiceIdleState(currentVoiceState)) {
        setTimeout(() => {
          try { wakeWordRecognizer.start(); } catch (e) {}
        }, 500);
      }
    };

    wakeWordRecognizer.start();
  } catch (err) {
    console.log("Wake word initialization notice:", err);
  }
}

async function sendAudioToBackend(audioBlob, isAssessment = false) {
  if (currentVoiceState === VoiceState.WAKE_ONLY || currentVoiceState === VoiceState.STANDBY) {
    console.log("[Voice SM] In STANDBY mode. Discarding audio upload.");
    return;
  }

  const recordStatus = document.getElementById("record-status-text");
  const timerEl = document.getElementById("record-hud-timer");

  const ext = audioBlob.type.includes("webm") ? "webm" : (audioBlob.type.includes("wav") ? "wav" : "ogg");
  const fileName = `browser_speech.${ext}`;

  currentAbortController = new AbortController();
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);
  formData.append("student_id", currentStudentId.toString());
  if (activeSessionId) formData.append("session_id", activeSessionId);
  formData.append("mode", "coach");
  const isAssess = isAssessment || (currentVoiceState === VoiceState.ASSESSMENT_RECORDING);
  if (isAssess) formData.append("is_assessment", "true");

  setVoiceState(VoiceState.PROCESSING, "sending_audio");

  try {
    if (recordStatus) recordStatus.textContent = "⚡ Transcribing speech with Whisper STT...";
    const res = await fetch("/api/v1/esp32/audio?format=json", {
      method: "POST",
      body: formData,
      signal: currentAbortController.signal
    });
    const data = await res.json();
    currentAbortController = null;
    isProcessingBackend = false;

    if (currentVoiceState === VoiceState.WAKE_ONLY || currentVoiceState === VoiceState.STANDBY || currentVoiceState === VoiceState.SNOOZED) {
      console.log("[Voice SM] Late response discarded due to stop command.");
      return;
    }

    if (data.is_control_command && data.command === "stop") {
      executeMikazaStop();
      return;
    }

    if (recordStatus) recordStatus.textContent = "Ready";
    if (timerEl) timerEl.textContent = "00:00 / 01:00";

    isAssessmentFinalizing = false;
    handleInteractionResponse(data);
  } catch (err) {
    isProcessingBackend = false;
    isAssessmentFinalizing = false;
    if (err.name === "AbortError") {
      console.log("[Voice SM] Audio upload aborted by stop command.");
      return;
    }
    currentAbortController = null;
    if (recordStatus) recordStatus.textContent = "Error processing audio";
    setSilenceBadge("Error • Ready", "normal");
    console.error(err);

    if (currentVoiceState !== VoiceState.WAKE_ONLY && currentVoiceState !== VoiceState.STANDBY && currentVoiceState !== VoiceState.SNOOZED) {
      setTimeout(() => {
        setVoiceState(VoiceState.READY, "error_resume");
        if (!isRecording && !isMicMuted) {
          startVoiceRecording(true);
        }
      }, 1000);
    }
  }
}

async function sendTextMessage(overrideText = null) {
  const textInput = document.getElementById("sim-text-input");
  const msg = (overrideText || (textInput ? textInput.value : "")).trim();
  if (!msg) return;
  if (!overrideText && textInput) textInput.value = "";

  // Check for "Mizo stop" typed command
  const lower = msg.toLowerCase();
  if (lower === "mizo stop" || lower === "stop mizo" || lower === "mikaza stop" || lower === "stop" || lower === "hey mizo stop" || lower === "hey mikaza stop") {
    executeMikazaStop();
    return;
  }

  currentAbortController = new AbortController();
  isProcessingBackend = true;
  setVoiceState(VoiceState.PROCESSING, "sending_text");

  try {
    const res = await fetch("/api/v1/esp32/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        student_id: currentStudentId,
        session_id: activeSessionId,
        mode: "coach",
        is_typed_text: true
      }),
      signal: currentAbortController.signal
    });
    const data = await res.json();
    currentAbortController = null;
    isProcessingBackend = false;

    if (currentVoiceState === VoiceState.WAKE_ONLY || currentVoiceState === VoiceState.STANDBY || currentVoiceState === VoiceState.SNOOZED) {
      console.log("[Voice SM] Late response discarded due to stop command.");
      return;
    }

    handleInteractionResponse(data);
  } catch (err) {
    isProcessingBackend = false;
    if (err.name === "AbortError") {
      console.log("[Voice SM] Text request aborted by stop command.");
      return;
    }
    currentAbortController = null;
    console.error("Chat error:", err);
    if (currentVoiceState !== VoiceState.WAKE_ONLY && currentVoiceState !== VoiceState.STANDBY && currentVoiceState !== VoiceState.SNOOZED) {
      setVoiceState(VoiceState.READY, "chat_error_fallback");
    }
  }
}

async function handleInteractionResponse(data) {
  // 1. Play Audio through Virtual Player
  if (data.audio_url) {
    const player = document.getElementById("sim-audio-player");
    const waveform = document.getElementById("sim-waveform");

    if (player) {
      isMizoSpeaking = true;
      isMikazaSpeaking = true;
      setSilenceBadge("🔊 Mizo speaking...", "playing");

      player.src = data.audio_url;
      player.style.display = "block";

      if (waveform) waveform.classList.add("playing");
      
      player.play().catch(e => {
        console.warn("Audio auto-play prevented:", e);
        isMizoSpeaking = false;
        isMikazaSpeaking = false;
        if (waveform) waveform.classList.remove("playing");
        setVoiceState(VoiceState.ACTIVE_LISTENING, "audio_play_prevented");
        if (!isRecording && !isMicMuted) {
          setTimeout(() => startVoiceRecording(true), 500);
        }
      });

      player.onended = () => {
        if (waveform) waveform.classList.remove("playing");
        isMizoSpeaking = false;
        isMikazaSpeaking = false;

        if (data.onboarding_step === "speech_test_prompt" || data.onboarding_step === "speaking_assessment") {
          console.log("[Voice SM] Transitioning to ASSESSMENT_RECORDING after Mizo finishes speaking assessment prompt.");
          setVoiceState(VoiceState.ASSESSMENT_RECORDING, "assessment_prompt_finished");
          showAssessmentHUD(data.topic || "1-Minute Speaking Sample");
          if (!isRecording && !isMicMuted) {
            setTimeout(() => startVoiceRecording(true), 400);
          }
          return;
        }

        setVoiceState(VoiceState.ACTIVE_LISTENING, "speech_playback_ended");

        // AUTOMATIC MICROPHONE AFTER EVERY QUESTION:
        // Whenever Mizo asks the user a question, the microphone automatically becomes ready for the user's response!
        // Do not require the user to say "Hey Mizo" after every question.
        if (!isRecording && !isMicMuted) {
          setTimeout(() => {
            if (!isRecording && !isMizoSpeaking && !isMikazaSpeaking && !isMicMuted && currentVoiceState === VoiceState.ACTIVE_LISTENING) {
              console.log("[Voice System] Question speech playback ended. Auto-activating mic for student reply!");
              startVoiceRecording(true);
            }
          }, 350);
        }
      };
    }
  } else {
    // If no audio was returned, clear processing state and resume listening if mic active
    if (data.onboarding_step === "speech_test_prompt" || data.onboarding_step === "speaking_assessment") {
      setVoiceState(VoiceState.ASSESSMENT_RECORDING, "assessment_prompt_no_audio");
      showAssessmentHUD(data.topic || "1-Minute Speaking Sample");
      if (!isRecording && !isMicMuted) {
        setTimeout(() => startVoiceRecording(true), 500);
      }
    } else {
      setVoiceState(VoiceState.READY, "no_audio_ready");
      setSilenceBadge("Ready", "idle");
      if (!isRecording && !isMicMuted) {
        setVoiceState(VoiceState.ACTIVE_LISTENING, "no_audio_auto_listen");
        setTimeout(() => startVoiceRecording(true), 600);
      }
    }
  }

  // 2. Update Session Metrics & Cumulative Progress
  if (data.session_metrics) {
    updateSessionBar("session-fluency", data.session_metrics.fluency_score, 1);
    updateSessionBar("session-grammar", data.session_metrics.grammar_accuracy, 1);
    updateSessionBar("session-vocab", data.session_metrics.vocabulary_richness, 1);
    updateSessionBar("session-pacing", data.session_metrics.pacing_score, 1);
  }

  if (data.historical_metrics) {
    await loadHistoricalProgress();
  }

  // 3. Check if an assessment was generated
  if (data.assessment) {
    renderLatestAssessmentPanel(data.assessment);
  }

  // 4. Reload conversation history, mistakes, and student profile
  loadConversations();
  loadStudentMistakes();
  loadStudents();
  loadAssessments();

  // 5. Automatic section switching from Agent Router
  if (data.active_section) {
    switchWorkspace(data.active_section);
  }
  if (data.active_section === "study" && data.response) {
    appendStudyMessage("mizo", data.response);
  }
}

function renderLatestAssessmentPanel(a) {
  const panel = document.getElementById("latest-assessment-panel");
  const header = document.getElementById("assessment-summary-header");
  const content = document.getElementById("latest-assessment-content");
  if (!panel || !content) return;

  panel.style.display = "block";
  if (header) {
    const scoreVal = (a.communication_score !== null && a.communication_score !== undefined) ? `${Math.round(a.communication_score)}%` : 'Pending';
    const methodBadge = a.assessment_method === 'llm'
      ? ` [AI - ${a.provider || 'LLM'}]`
      : (a.assessment_method === 'fallback' ? ` [Deterministic Fallback]` : '');
    header.textContent = `Baseline Assessment Completed: ${a.overall_level || 'Assessed'} (${scoreVal} Communication Score)${methodBadge}`;
  }

  const methodLabel = a.assessment_method ? a.assessment_method.toUpperCase() : 'LLM';
  const qualityLabel = a.assessment_quality || 'normal';

  content.innerHTML = `
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;">
      <div><span class="text-muted">Grammar:</span> <strong>${a.grammar_score != null ? Math.round(a.grammar_score) + '%' : 'N/A'}</strong></div>
      <div><span class="text-muted">Vocabulary:</span> <strong>${a.vocabulary_score != null ? Math.round(a.vocabulary_score) + '%' : 'N/A'}</strong></div>
      <div><span class="text-muted">Fluency:</span> <strong>${a.fluency_score != null ? Math.round(a.fluency_score) + '%' : 'N/A'}</strong></div>
      <div><span class="text-muted">Pronunciation:</span> <strong>${a.pronunciation_score != null ? Math.round(a.pronunciation_score) + '%' : 'N/A'}</strong></div>
      <div><span class="text-muted">Confidence:</span> <strong>${a.confidence_score != null ? Math.round(a.confidence_score) + '%' : 'N/A'}</strong></div>
      <div><span class="text-muted">Method:</span> <strong>${escapeHtml(methodLabel)}</strong></div>
      <div><span class="text-muted">Quality:</span> <strong>${escapeHtml(qualityLabel)}</strong></div>
      <div><span class="text-muted">Overall:</span> <strong>${escapeHtml(a.overall_level || 'Assessed')}</strong></div>
    </div>

    <div style="line-height:1.5;">
      <div><strong>Grammar Assessment:</strong> <span class="text-muted">${escapeHtml(a.grammar_feedback)}</span></div>
      <div><strong>Vocabulary Breadth:</strong> <span class="text-muted">${escapeHtml(a.vocabulary_feedback)}</span></div>
      <div><strong>Fluency & Flow:</strong> <span class="text-muted">${escapeHtml(a.fluency_feedback)}</span></div>
      <div><strong>Pronunciation:</strong> <span class="text-muted">${escapeHtml(a.pronunciation_feedback)}</span></div>
      <div style="font-size:0.75rem; color:var(--text-dim); margin-top:2px;"><em>Note: Pronunciation evaluated from speech recognition intelligibility and transcript flow. Acoustic phonetics are not measured directly from transcript alone.</em></div>
      <div style="margin-top:6px;"><strong>Delivery & Confidence:</strong> <span class="text-muted">${escapeHtml(a.confidence_feedback)}</span></div>
    </div>

    <div style="display:flex; gap:16px; flex-wrap:wrap; font-size:0.85rem; padding-top:6px; border-top:1px solid rgba(255,255,255,0.06);">
      <span style="color:#34d399;"><strong>Identified Strengths:</strong> ${(a.strengths || []).join(", ") || "Active participation"}</span>
      <span style="color:#f43f5e;"><strong>Focus Challenges:</strong> ${(a.weaknesses || []).join(", ") || "None"}</span>
    </div>
  `;
}

// -------------------------------------------------------------
// Event Listeners & Core Actions
// -------------------------------------------------------------
function setupEventListeners() {
  setupStudentProfileModal();
  setupDeleteLearnerModal();
  setupShortRecordingModal();
  setupStudySectionEventListeners();
  setupSpeechSectionEventListeners();

  // 1. Start New Session Button
  const btnNewSession = document.getElementById("btn-global-new-session");
  if (btnNewSession) {
    btnNewSession.addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ student_id: currentStudentId, mode: "coach" })
        });
        if (res.ok) {
          const session = await res.json();
          activeSessionId = session.session_id;
          showToast("New Learning Session started! Profile & history preserved.");
          await loadActiveSession();
          await loadStudents();
        }
      } catch (e) {
        showToast("Error starting new session", true);
      }
    });
  }

  // 1b. Reset Current Session Button (Part 23)
  const btnResetSession = document.getElementById("btn-reset-session");
  if (btnResetSession) {
    btnResetSession.addEventListener("click", async () => {
      if (!confirm("Reset Current Session?\n\nThis will clear current session metrics and turns without deleting historical progress or your learner profile.")) {
        return;
      }
      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}/sessions/reset-current`, {
          method: "POST"
        });
        if (res.ok) {
          showToast("Current session reset! Session metrics cleared, historical progress preserved.");
          await loadActiveSession();
          await loadConversations();
          await loadHistoricalProgress();
        } else {
          showToast("Error resetting current session", true);
        }
      } catch (err) {
        showToast("Network error resetting current session", true);
      }
    });
  }

  // 2. Restart Onboarding Button
  const btnRestartOnboarding = document.getElementById("btn-restart-onboarding");
  if (btnRestartOnboarding) {
    btnRestartOnboarding.addEventListener("click", async () => {
      if (!confirm("Restart Onboarding for a fresh run?\n\nThis will reset your profile back to Step 1 (Name greeting) and start fresh.")) {
        return;
      }
      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}/restart-onboarding`, {
          method: "POST"
        });
        if (res.ok) {
          showToast("Onboarding restarted from Step 1!");
          await loadAllData();

          // Switch to Onboarding & Voice Chat Tab
          const simTabBtn = document.getElementById("nav-simulator-btn");
          if (simTabBtn) simTabBtn.click();

          // Reset dialogue display
          const chatBox = document.getElementById("chat-messages-container");
          if (chatBox) {
            chatBox.innerHTML = `
              <div class="chat-bubble assistant">
                <div class="chat-avatar">🤖</div>
                <div class="chat-body">
                  <div class="chat-meta"><strong>Mizo AI</strong> • Just now</div>
                  <div class="chat-text">Hello! I'm Mizo, your AI English learning robot tutor. To get started, what should I call you?</div>
                </div>
              </div>
            `;
          }

          // Trigger Mizo spoken greeting & open mic
          await triggerMizoStartGreeting();
        } else {
          showToast("Error restarting onboarding", true);
        }
      } catch (err) {
        showToast("Network error restarting onboarding", true);
      }
    });
  }

  // 3. Skip Audio Onboarding Button
  const btnSkipAudio = document.getElementById("btn-skip-audio-onboarding");
  if (btnSkipAudio) {
    btnSkipAudio.addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}/complete-text-onboarding`, {
          method: "POST"
        });
        if (res.ok) {
          showToast("Text profile completed! Speaking assessment marked pending.");
          await loadAllData();
        } else {
          showToast("Error completing text profile", true);
        }
      } catch (e) {
        showToast("Network error completing text profile", true);
      }
    });
  }

  // 4. Conversation Search Input
  const inputSearch = document.getElementById("input-history-search");
  if (inputSearch) {
    let debounceTimer = null;
    inputSearch.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        loadConversations(inputSearch.value);
      }, 300);
    });
  }

  // 5. Refresh Buttons
  const btnRefreshData = document.getElementById("btn-refresh-data");
  if (btnRefreshData) {
    btnRefreshData.addEventListener("click", () => {
      loadAllData();
      showToast("All dashboard data refreshed");
    });
  }
  const btnRefreshHistory = document.getElementById("btn-refresh-history");
  if (btnRefreshHistory) {
    btnRefreshHistory.addEventListener("click", () => {
      loadConversations();
      showToast("Conversation history refreshed");
    });
  }

  // 6. Test Providers
  const btnTestProviders = document.getElementById("btn-test-providers");
  if (btnTestProviders) {
    btnTestProviders.addEventListener("click", async () => {
      showToast("Testing AI provider connectivity...");
      await loadProviderStatus();
      showToast("Provider connectivity checked");
    });
  }

  // 7. Add Device
  const btnAddDevice = document.getElementById("btn-add-device");
  if (btnAddDevice) {
    btnAddDevice.addEventListener("click", async () => {
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
  }

  // 12. PDF Dropzone
  const dropzone = document.getElementById("pdf-dropzone");
  const fileInput = document.getElementById("pdf-file-input");
  if (dropzone && fileInput) {
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
}

// -------------------------------------------------------------
// Delete Learner Data Modal Handler
// -------------------------------------------------------------
function setupDeleteLearnerModal() {
  const modal = document.getElementById("modal-delete-confirm");
  const btnOpen = document.getElementById("btn-delete-learner-data");
  const btnClose = document.getElementById("btn-close-delete-modal");
  const btnCancel = document.getElementById("btn-cancel-delete");
  const btnConfirm = document.getElementById("btn-confirm-delete");

  if (!modal || !btnOpen) return;

  const openModal = () => { modal.style.display = "flex"; };
  const closeModal = () => { modal.style.display = "none"; };

  btnOpen.addEventListener("click", openModal);
  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (btnCancel) btnCancel.addEventListener("click", closeModal);

  if (btnConfirm) {
    btnConfirm.addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}/data`, {
          method: "DELETE"
        });
        if (res.ok) {
          closeModal();
          showToast("Learner profile & history deleted! Starting fresh onboarding.");
          await loadAllData();

          // Switch to Onboarding Tab
          const simTabBtn = document.getElementById("nav-simulator-btn");
          if (simTabBtn) simTabBtn.click();

          // Reset dialogue display
          const chatBox = document.getElementById("chat-messages-container");
          if (chatBox) {
            chatBox.innerHTML = `
              <div class="chat-bubble assistant">
                <div class="chat-avatar">🤖</div>
                <div class="chat-body">
                  <div class="chat-meta"><strong>Mizo AI</strong> • Just now</div>
                  <div class="chat-text">Hello! I'm Mizo, your AI English learning robot tutor. To get started, what should I call you?</div>
                </div>
              </div>
            `;
          }

          // Trigger Mizo fresh greeting & voice start
          await triggerMizoStartGreeting();
        } else {
          showToast("Error deleting learner data", true);
        }
      } catch (err) {
        showToast("Network error deleting learner data", true);
      }
    });
  }
}

// -------------------------------------------------------------
// Short Audio Recording Warning Modal Handler
// -------------------------------------------------------------
function setupShortRecordingModal() {
  const modal = document.getElementById("modal-short-recording");
  const btnClose = document.getElementById("btn-close-short-modal");
  const btnContinue = document.getElementById("btn-continue-recording");
  const btnSubmitAnyway = document.getElementById("btn-submit-anyway");

  if (!modal) return;

  const closeModal = () => { modal.style.display = "none"; };

  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (btnContinue) {
    btnContinue.addEventListener("click", () => {
      closeModal();
      showToast("Ready to re-record speech sample");
    });
  }
  if (btnSubmitAnyway) {
    btnSubmitAnyway.addEventListener("click", () => {
      closeModal();
      if (lastRecordedBlob) {
        sendAudioToBackend(lastRecordedBlob);
      }
    });
  }
}

// -------------------------------------------------------------
// Student Profile Edit Modal
// -------------------------------------------------------------
function setupStudentProfileModal() {
  const modal = document.getElementById("modal-edit-profile");
  const btnOpen = document.getElementById("btn-edit-student-modal");
  const btnClose = document.getElementById("btn-close-student-modal");
  const btnCancel = document.getElementById("btn-cancel-student-edit");
  const btnSave = document.getElementById("btn-save-student-profile");

  if (!modal || !btnOpen) return;

  const openModal = async () => {
    try {
      const res = await fetch(`/api/v1/students/${currentStudentId}`);
      if (res.ok) {
        const s = await res.json();
        document.getElementById("edit-student-name").value = s.name || "";
        document.getElementById("edit-student-grade").value = s.education || s.grade || "";
        document.getElementById("edit-student-target").value = s.self_reported_level || "Not sure";
        document.getElementById("edit-student-goals").value = s.learning_goals || "";
        document.getElementById("edit-student-topics").value = s.learning_topics || s.interests || "";
        document.getElementById("edit-student-strengths").value = (s.strengths || []).join(", ");
        document.getElementById("edit-student-weaknesses").value = (s.weaknesses || []).join(", ");
      }
    } catch (e) {
      console.error("Error loading student profile:", e);
    }
    modal.style.display = "flex";
  };

  const closeModal = () => { modal.style.display = "none"; };

  btnOpen.addEventListener("click", openModal);
  if (btnClose) btnClose.addEventListener("click", closeModal);
  if (btnCancel) btnCancel.addEventListener("click", closeModal);

  if (btnSave) {
    btnSave.addEventListener("click", async () => {
      const name = document.getElementById("edit-student-name").value.trim();
      const grade = document.getElementById("edit-student-grade").value.trim();
      const target_level = document.getElementById("edit-student-target").value;
      const learning_goals = document.getElementById("edit-student-goals").value.trim();
      const learning_topics = document.getElementById("edit-student-topics").value.trim();
      const rawWeak = document.getElementById("edit-student-weaknesses").value;
      const rawStrong = document.getElementById("edit-student-strengths").value;

      const weaknesses = rawWeak.split(",").map(w => w.trim()).filter(w => w.length > 0);
      const strengths = rawStrong.split(",").map(s => s.trim()).filter(s => s.length > 0);

      const payload = {
        name: name || "Student",
        education: grade || "Student",
        grade: grade || "Student",
        self_reported_level: target_level,
        learning_goals: learning_goals,
        learning_topics: learning_topics,
        interests: learning_topics,
        weaknesses: weaknesses,
        strengths: strengths
      };

      try {
        const res = await fetch(`/api/v1/students/${currentStudentId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          showToast("Student profile updated successfully!");
          closeModal();
          await loadStudents();
          await loadActiveSession();
        } else {
          showToast("Failed to save profile updates", true);
        }
      } catch (err) {
        showToast("Error updating student profile", true);
      }
    });
  }
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
      await loadSettings();
      if (payload.groq_api_key) {
        testSingleProvider("groq");
      } else if (payload.openai_api_key) {
        testSingleProvider("openai");
      } else if (payload.qwen_api_key) {
        testSingleProvider("qwen");
      } else if (activeProvider) {
        testSingleProvider(activeProvider);
      }
      loadProviderStatus();
    } else {
      showToast("Error saving settings", true);
    }
  } catch (e) {
    showToast("Network error saving settings", true);
  }
}

// -------------------------------------------------------------
// Settings & Custom Prompt Event Handlers
// -------------------------------------------------------------
function setupSettingsEventHandlers() {
  // Save Behavior & Personalized System Prompt
  const btnSaveBehavior = document.getElementById("btn-save-behavior");
  if (btnSaveBehavior) {
    btnSaveBehavior.addEventListener("click", async () => {
      const systemPrompt = document.getElementById("setting-system-prompt")?.value || "";
      const coachingMode = document.getElementById("setting-coaching-mode")?.value || "coach";
      const strictness = document.getElementById("setting-grammar-strictness")?.value || "balanced";
      const ttsVoice = document.getElementById("setting-tts-voice")?.value || "en-US-GuyNeural";
      const ttsRate = document.getElementById("setting-tts-rate")?.value || "+0%";

      await saveSettingsAPI({
        system_prompt: systemPrompt,
        coaching_mode: coachingMode,
        grammar_strictness: strictness,
        tts_voice: ttsVoice,
        tts_rate: ttsRate
      });
    });
  }

  // Save AI Providers & API Keys
  const btnSaveProviders = document.getElementById("btn-save-providers");
  if (btnSaveProviders) {
    btnSaveProviders.addEventListener("click", async () => {
      const groqModel = document.getElementById("input-groq-model")?.value || "openai/gpt-oss-120b";
      const groqKey = document.getElementById("input-groq-key")?.value || "";
      const openaiModel = document.getElementById("input-openai-model")?.value || "gpt-4o-mini";
      const openaiKey = document.getElementById("input-openai-key")?.value || "";
      const qwenModel = document.getElementById("input-qwen-model")?.value || "qwen/qwen-2.5-72b-instruct";
      const qwenKey = document.getElementById("input-qwen-key")?.value || "";
      const ollamaUrl = document.getElementById("input-ollama-url")?.value || "http://localhost:11434";
      const ollamaModel = document.getElementById("input-ollama-model")?.value || "llama3:latest";

      const payload = {
        active_provider: activeProvider,
        groq_model: groqModel.trim(),
        openai_model: openaiModel.trim(),
        qwen_model: qwenModel.trim(),
        ollama_base_url: ollamaUrl.trim(),
        ollama_model: ollamaModel.trim()
      };

      if (groqKey.trim()) payload.groq_api_key = groqKey.trim();
      if (openaiKey.trim()) payload.openai_api_key = openaiKey.trim();
      if (qwenKey.trim()) payload.qwen_api_key = qwenKey.trim();

      await saveSettingsAPI(payload);
    });
  }

  // Refresh Groq Models button
  const btnRefreshGroq = document.getElementById("btn-refresh-groq-models");
  if (btnRefreshGroq) {
    btnRefreshGroq.addEventListener("click", async () => {
      btnRefreshGroq.disabled = true;
      const origText = btnRefreshGroq.textContent;
      btnRefreshGroq.textContent = "⏳...";
      await fetchProviderModels("groq");
      btnRefreshGroq.textContent = origText;
      btnRefreshGroq.disabled = false;
      showToast("Groq model catalog refreshed.");
    });
  }

  // Individual Provider Test Buttons
  document.querySelectorAll(".btn-test-single-key").forEach(btn => {
    btn.addEventListener("click", async () => {
      const prov = btn.getAttribute("data-provider");
      if (!prov) return;
      const input = document.getElementById(`input-${prov}-key`);
      const typedKey = input ? input.value.trim() : "";
      if (typedKey && !typedKey.includes("•")) {
        // Persist newly typed key into canonical storage first, then test
        const payload = { active_provider: activeProvider };
        payload[`${prov}_api_key`] = typedKey;
        await saveSettingsAPI(payload);
      } else {
        await testSingleProvider(prov);
      }
    });
  });

  // Clear API Key Buttons
  document.querySelectorAll(".btn-clear-key").forEach(btn => {
    btn.addEventListener("click", async () => {
      const prov = btn.getAttribute("data-provider");
      const targetId = btn.getAttribute("data-target");
      if (!prov) return;
      if (confirm(`Are you sure you want to delete and clear the stored ${formatProviderName(prov)} API key?`)) {
        const payload = {};
        payload[`clear_${prov}_key`] = true;
        await saveSettingsAPI(payload);
        const input = document.getElementById(targetId);
        if (input) input.value = "";
        showToast(`${formatProviderName(prov)} API key cleared.`);
        await loadSettings();
        updateProviderUIStatus(prov, { configured: false, status: "not_configured", provider: prov });
      }
    });
  });

  // Toggle Password Visibility
  document.querySelectorAll(".btn-toggle-key").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (input) {
        input.type = input.type === "password" ? "text" : "password";
        btn.textContent = input.type === "password" ? "👁️" : "🙈";
      }
    });
  });

  // Provider Card Selection
  document.querySelectorAll(".provider-card").forEach(card => {
    card.addEventListener("click", () => {
      const prov = card.getAttribute("data-provider");
      if (prov) {
        activeProvider = prov;
        document.querySelectorAll(".provider-card").forEach(c => {
          c.classList.toggle("active", c.getAttribute("data-provider") === activeProvider);
        });
        const pillText = card.querySelector(".card-status-pill")?.textContent || "";
        if (pillText && !pillText.includes("READY")) {
          showToast(`Selected primary: ${formatProviderName(activeProvider)} (${pillText} — Mizo will cascade to ready fallbacks)`, true);
        } else {
          showToast(`Selected primary provider: ${formatProviderName(activeProvider)}`);
        }
      }
    });
  });

  // Test All Provider Connections
  const btnTestProviders = document.getElementById("btn-test-providers");
  if (btnTestProviders) {
    btnTestProviders.addEventListener("click", async () => {
      const container = document.getElementById("provider-status-container");
      if (container) {
        container.innerHTML = `
          <div class="provider-status-card" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; display:flex; justify-content:center; align-items:center;">
            <span style="color:#818cf8;">⚡ Testing all AI provider connections in parallel...</span>
          </div>
        `;
      }
      showToast("Testing provider connections...");
      await loadProviderStatus();
    });
  }
}

// -------------------------------------------------------------
// Utilities
// -------------------------------------------------------------
function showToast(msg, isError = false) {
  const toast = document.getElementById("toast");
  if (!toast) return;
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

// -------------------------------------------------------------
// Workspace Switching (Agent Router Automatic Navigation)
// -------------------------------------------------------------
function switchWorkspace(section) {
  let tabId = "tab-simulator";
  if (section === "study") tabId = "tab-study";
  else if (section === "speech") tabId = "tab-speech";
  else if (section === "normal_chat") tabId = "tab-simulator";
  else if (section && section.startsWith("tab-")) tabId = section;

  const targetTab = document.querySelector(`.nav-item[data-tab="${tabId}"]`);
  if (targetTab && !targetTab.classList.contains("active")) {
    console.log(`[Workspace Router] Switching workspace to: ${section} (${tabId})`);
    targetTab.click();
  }
}

// -------------------------------------------------------------
// Study / Material-Based Teaching Section
// -------------------------------------------------------------
let activeSubjectId = 1;
let currentStudyTopic = null;

async function loadStudyWorkspace() {
  await Promise.all([
    loadStudySubjects(),
    loadStudyProgress()
  ]);
}

async function loadStudySubjects() {
  try {
    const res = await fetch("/api/v1/subjects");
    if (!res.ok) return;
    const subjects = await res.json();
    const select = document.getElementById("study-subject-select");
    const badge = document.getElementById("study-active-subject-badge");
    
    if (select && subjects.length > 0) {
      select.innerHTML = subjects.map(s => `<option value="${s.id}">${escapeHtml(s.name)}</option>`).join("");
      activeSubjectId = subjects[0].id;
      if (badge) badge.textContent = `Subject: ${subjects[0].name}`;
      await loadSubjectSyllabus(activeSubjectId);
    }
  } catch (e) {
    console.warn("Failed to load study subjects:", e);
  }
}

async function loadSubjectSyllabus(subId) {
  const treeBox = document.getElementById("study-syllabus-tree");
  if (!treeBox) return;

  try {
    const res = await fetch(`/api/v1/subjects/${subId}`);
    if (!res.ok) {
      treeBox.innerHTML = `<div class="text-muted" style="text-align:center; padding:15px; font-size:0.82rem;">No syllabus units loaded yet. Drop a syllabus or notes on the left.</div>`;
      return;
    }
    const data = await res.json();
    const units = data.units || [];
    if (units.length === 0) {
      treeBox.innerHTML = `<div class="text-muted" style="text-align:center; padding:15px; font-size:0.82rem;">No units in syllabus. Upload material to index topics.</div>`;
      return;
    }

    let html = "";
    units.forEach((unit, uIdx) => {
      html += `
        <div class="syllabus-unit-group" style="margin-bottom:12px;">
          <div style="font-weight:600; font-size:0.82rem; color:var(--text-accent); margin-bottom:6px;">
            Unit ${unit.unit_order || (uIdx + 1)}: ${escapeHtml(unit.title)}
          </div>
          <div style="display:flex; flex-direction:column; gap:4px; padding-left:8px;">
      `;
      (unit.topics || []).forEach(t => {
        const isCurrent = currentStudyTopic && currentStudyTopic.toLowerCase() === t.title.toLowerCase();
        html += `
          <button class="syllabus-topic-item ${isCurrent ? 'active' : ''}" data-topic-title="${escapeHtml(t.title)}" style="text-align:left; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:6px 10px; font-size:0.8rem; cursor:pointer; color:var(--text-light); transition:all 0.2s ease;">
            📘 ${escapeHtml(t.title)}
          </button>
        `;
      });
      html += `</div></div>`;
    });
    treeBox.innerHTML = html;

    // Attach click listeners to syllabus topics
    treeBox.querySelectorAll(".syllabus-topic-item").forEach(item => {
      item.addEventListener("click", async () => {
        const topicTitle = item.getAttribute("data-topic-title");
        await selectStudyTopic(topicTitle);
      });
    });
  } catch (e) {
    console.warn("Failed to load subject syllabus:", e);
    treeBox.innerHTML = `<div class="text-muted" style="text-align:center; padding:15px;">Error loading syllabus.</div>`;
  }
}

async function selectStudyTopic(topicTitle) {
  currentStudyTopic = topicTitle;
  const header = document.getElementById("study-current-topic-title");
  if (header) header.textContent = `Current Topic: ${topicTitle}`;

  // Highlight in syllabus tree
  document.querySelectorAll(".syllabus-topic-item").forEach(b => {
    b.classList.toggle("active", b.getAttribute("data-topic-title") === topicTitle);
  });

  try {
    await fetch(`/api/v1/subjects/${activeSubjectId}/select?student_id=${currentStudentId}`, {
      method: "POST"
    });
  } catch (e) {}

  appendStudyMessage("user", `Teach me ${topicTitle}`);
  await sendStudyMessageDirect(`Teach me ${topicTitle}`);
}

async function loadStudyProgress() {
  const container = document.getElementById("study-progress-container");
  if (!container) return;

  try {
    const res = await fetch(`/api/v1/subjects/${activeSubjectId}/progress?student_id=${currentStudentId}`);
    if (!res.ok) return;
    const prog = await res.json();

    const weakTopics = (prog.weak_topics_json ? JSON.parse(prog.weak_topics_json) : []) || [];
    let html = `
      <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
        <span>Subject Mastery:</span>
        <strong style="color:var(--accent-glow);">${prog.mastery_score || 0}%</strong>
      </div>
      <div class="progress-bar-bg" style="height:6px; background:rgba(255,255,255,0.08); border-radius:3px; margin-bottom:14px; overflow:hidden;">
        <div style="height:100%; width:${Math.min(100, Math.max(0, prog.mastery_score || 0))}%; background:var(--accent-glow);"></div>
      </div>
    `;

    if (weakTopics.length > 0) {
      html += `
        <div style="margin-top:10px;">
          <div style="color:#f43f5e; font-weight:600; font-size:0.8rem; margin-bottom:6px;">⚠️ Weak Topics Requiring Focus:</div>
          <div style="display:flex; flex-wrap:wrap; gap:6px;">
            ${weakTopics.map(w => `<span class="tag tag-danger" style="font-size:0.75rem;">${escapeHtml(w)}</span>`).join("")}
          </div>
        </div>
      `;
    } else {
      html += `<div class="text-muted" style="font-size:0.78rem;">No weak topics flagged yet. Answer quizzes to identify challenge areas.</div>`;
    }

    container.innerHTML = html;
  } catch (e) {
    console.warn("Failed to load study progress:", e);
  }
}

function appendStudyMessage(role, text) {
  const container = document.getElementById("study-messages-container");
  if (!container) return;

  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${role === "user" ? "user-msg" : "mizo-msg"}`;
  msgDiv.style.marginBottom = "10px";
  msgDiv.style.display = "flex";
  msgDiv.style.flexDirection = "column";
  msgDiv.style.alignItems = role === "user" ? "flex-end" : "flex-start";

  msgDiv.innerHTML = `
    <div style="max-width:85%; background:${role === 'user' ? 'rgba(99, 102, 241, 0.2)' : 'rgba(255, 255, 255, 0.05)'}; border:1px solid ${role === 'user' ? 'rgba(99, 102, 241, 0.4)' : 'rgba(255, 255, 255, 0.1)'}; border-radius:10px; padding:10px 14px; font-size:0.88rem; line-height:1.45;">
      <div style="font-size:0.7rem; font-weight:600; color:var(--text-muted); margin-bottom:4px;">
        ${role === 'user' ? 'You' : '📚 Study Tutor'}
      </div>
      <div>${escapeHtml(text)}</div>
    </div>
  `;

  // Remove placeholder if present
  if (container.querySelector(".text-muted")) {
    container.innerHTML = "";
  }

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
}

async function sendStudyMessageDirect(msg) {
  if (!msg) return;
  try {
    const res = await fetch("/api/v1/esp32/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        student_id: currentStudentId,
        session_id: activeSessionId,
        mode: "tutor",
        is_typed_text: true
      })
    });
    const data = await res.json();
    if (data.response) {
      appendStudyMessage("mizo", data.response);
    }
    loadStudyProgress();
  } catch (err) {
    console.error("Study chat error:", err);
  }
}

function setupStudySectionEventListeners() {
  // Quick Action pills
  document.querySelectorAll("[data-tutor-cmd]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const cmd = btn.getAttribute("data-tutor-cmd");
      appendStudyMessage("user", cmd);
      await sendStudyMessageDirect(cmd);
    });
  });

  // Study text input
  const input = document.getElementById("study-text-input");
  const sendBtn = document.getElementById("btn-study-send");
  if (sendBtn && input) {
    const sendHandler = async () => {
      const val = input.value.trim();
      if (!val) return;
      input.value = "";
      appendStudyMessage("user", val);
      await sendStudyMessageDirect(val);
    };
    sendBtn.addEventListener("click", sendHandler);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") sendHandler();
    });
  }

  // Refresh button
  const refreshBtn = document.getElementById("btn-study-refresh");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", () => {
      loadStudyWorkspace();
      showToast("Study material refreshed!");
    });
  }

  // Clear / New Study Session button
  const clearSessionBtn = document.getElementById("btn-study-clear-session");
  if (clearSessionBtn) {
    clearSessionBtn.addEventListener("click", async () => {
      try {
        await fetch(`/api/v1/subjects/session/reset?student_id=${currentStudentId}`, { method: "POST" });
        const container = document.getElementById("study-messages-container");
        if (container) {
          container.innerHTML = `<div class="text-muted" style="text-align:center; padding:30px;">📚 Fresh study session started! Ask any concept from your syllabus.</div>`;
        }
        showToast("New Study Session started! Old tutor context cleared.");
      } catch (e) {
        showToast("Error resetting study session", true);
      }
    });
  }

  // Subject select dropdown
  const subjectSelect = document.getElementById("study-subject-select");
  if (subjectSelect) {
    subjectSelect.addEventListener("change", async (e) => {
      activeSubjectId = parseInt(e.target.value, 10);
      const opt = e.target.options[e.target.selectedIndex];
      const badge = document.getElementById("study-active-subject-badge");
      if (badge && opt) badge.textContent = `Subject: ${opt.text}`;
      await loadSubjectSyllabus(activeSubjectId);
      await loadStudyProgress();
    });
  }

  // Material dropzone & file upload
  const dropzone = document.getElementById("study-pdf-dropzone");
  const fileInput = document.getElementById("study-file-input");
  const statusEl = document.getElementById("study-upload-status");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      if (statusEl) statusEl.innerHTML = `<span style="color:#818cf8;">⏳ Uploading and indexing ${escapeHtml(file.name)}...</span>`;

      try {
        if (file.name.endsWith(".json")) {
          const reader = new FileReader();
          reader.onload = async (evt) => {
            try {
              const syllabusJson = JSON.parse(evt.target.result);
              const res = await fetch("/api/v1/subjects/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(syllabusJson)
              });
              if (res.ok) {
                if (statusEl) statusEl.innerHTML = `<span style="color:#34d399;">✓ Syllabus loaded successfully!</span>`;
                showToast("Syllabus loaded successfully!");
                await loadStudyWorkspace();
              } else {
                throw new Error("Failed to load syllabus JSON");
              }
            } catch (err) {
              if (statusEl) statusEl.innerHTML = `<span style="color:#f43f5e;">Upload failed: ${err.message}</span>`;
            }
          };
          reader.readAsText(file);
        } else {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/v1/knowledge/upload", {
            method: "POST",
            body: formData
          });
          if (res.ok) {
            if (statusEl) statusEl.innerHTML = `<span style="color:#34d399;">✓ Study document uploaded & indexed!</span>`;
            showToast("Document indexed for study agent!");
          } else {
            throw new Error("Upload failed");
          }
        }
      } catch (err) {
        if (statusEl) statusEl.innerHTML = `<span style="color:#f43f5e;">Error uploading material</span>`;
      }
    });
  }
}

// -------------------------------------------------------------
// Speech / Seminar Practice Section
// -------------------------------------------------------------
let speechTimerInterval = null;
let speechElapsedSeconds = 0;
let speechAllowedSeconds = 180;
let speechTopic = "";
let speechRequiredPoints = [];

function loadSpeechWorkspace() {
  updateSpeechPointsPreview();
}

function updateSpeechPointsPreview() {
  const preview = document.getElementById("speech-points-preview");
  if (!preview) return;

  if (!speechTopic && speechRequiredPoints.length === 0) {
    preview.innerHTML = `<span class="text-muted">No topic configured yet. Enter topic and points above.</span>`;
    return;
  }

  let html = `
    <div style="font-weight:600; color:var(--text-light); margin-bottom:8px;">
      🎤 ${escapeHtml(speechTopic || "Presentation Topic")}
    </div>
    <div style="display:flex; flex-direction:column; gap:6px;">
  `;
  speechRequiredPoints.forEach((pt, idx) => {
    html += `
      <div style="display:flex; align-items:center; gap:8px; font-size:0.83rem;">
        <span style="color:var(--text-accent);">[${idx + 1}]</span>
        <span>${escapeHtml(pt)}</span>
      </div>
    `;
  });
  html += `</div>`;
  preview.innerHTML = html;
}

function formatDurationDisplay(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const s = (sec % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function setupSpeechSectionEventListeners() {
  const saveBtn = document.getElementById("btn-save-speech-setup");
  const topicInput = document.getElementById("speech-topic-input");
  const pointsInput = document.getElementById("speech-points-input");
  const durationSelect = document.getElementById("speech-duration-select");

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      speechTopic = topicInput ? topicInput.value.trim() : "";
      const rawPoints = pointsInput ? pointsInput.value.trim() : "";
      speechRequiredPoints = rawPoints ? rawPoints.split("\n").map(p => p.trim()).filter(Boolean) : [];
      speechAllowedSeconds = durationSelect ? parseInt(durationSelect.value, 10) : 180;

      if (!speechTopic) {
        showToast("Please enter a speech or presentation topic", true);
        return;
      }

      try {
        const res = await fetch("/api/v1/speech/setup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            student_id: currentStudentId,
            topic: speechTopic,
            required_points: speechRequiredPoints,
            allowed_duration_seconds: speechAllowedSeconds
          })
        });
        if (res.ok) {
          const badge = document.getElementById("speech-session-state-badge");
          if (badge) badge.textContent = "State: READY";
          const timerDisplay = document.getElementById("speech-live-timer");
          if (timerDisplay) timerDisplay.textContent = `00:00 / ${formatDurationDisplay(speechAllowedSeconds)}`;
          updateSpeechPointsPreview();
          showToast("Rehearsal stage configured! Ready to start.");
        }
      } catch (e) {
        showToast("Failed to save speech setup", true);
      }
    });
  }

  // Start Rehearsal
  const startBtn = document.getElementById("btn-speech-start");
  const finishBtn = document.getElementById("btn-speech-finish");
  const timerDisplay = document.getElementById("speech-live-timer");
  const hudBadge = document.getElementById("speech-hud-state");
  const hintEl = document.getElementById("speech-status-hint");

  if (startBtn && finishBtn) {
    startBtn.addEventListener("click", () => {
      if (speechTimerInterval) clearInterval(speechTimerInterval);
      speechElapsedSeconds = 0;

      if (hudBadge) {
        hudBadge.textContent = "RECORDING";
        hudBadge.style.background = "#f43f5e";
      }
      if (hintEl) hintEl.textContent = "Rehearsal in progress... Speak clearly into your microphone.";

      startBtn.disabled = true;
      finishBtn.disabled = false;

      speechTimerInterval = setInterval(() => {
        speechElapsedSeconds++;
        if (timerDisplay) {
          timerDisplay.textContent = `${formatDurationDisplay(speechElapsedSeconds)} / ${formatDurationDisplay(speechAllowedSeconds)}`;
        }
        if (speechElapsedSeconds >= speechAllowedSeconds) {
          if (hintEl) hintEl.textContent = "Time is up! Wrap up your speech and click Finish & Analyze.";
        }
      }, 1000);
    });

    finishBtn.addEventListener("click", async () => {
      if (speechTimerInterval) {
        clearInterval(speechTimerInterval);
        speechTimerInterval = null;
      }
      startBtn.disabled = false;
      finishBtn.disabled = true;

      if (hudBadge) {
        hudBadge.textContent = "ANALYZING";
        hudBadge.style.background = "#818cf8";
      }
      if (hintEl) hintEl.textContent = "Analyzing presentation delivery, rubric coverage, and pacing...";

      const transcriptEl = document.getElementById("speech-transcript-input");
      const transcript = transcriptEl && transcriptEl.value.trim()
        ? transcriptEl.value.trim()
        : `Today I will speak about ${speechTopic || "my topic"}. ${speechRequiredPoints.join(". ")}. In conclusion, this covers our main points.`;

      await analyzeSpeechRehearsal(transcript);
    });
  }

  // Analyze Transcript Button
  const analyzeBtn = document.getElementById("btn-analyze-transcript");
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      const transcriptEl = document.getElementById("speech-transcript-input");
      const transcript = transcriptEl ? transcriptEl.value.trim() : "";
      if (!transcript) {
        showToast("Please enter a speech transcript to analyze", true);
        return;
      }
      await analyzeSpeechRehearsal(transcript);
    });
  }

  // New Rehearsal Button
  const newBtn = document.getElementById("btn-speech-new");
  if (newBtn) {
    newBtn.addEventListener("click", () => {
      if (speechTimerInterval) clearInterval(speechTimerInterval);
      speechTimerInterval = null;
      speechElapsedSeconds = 0;
      speechTopic = "";
      speechRequiredPoints = [];

      if (topicInput) topicInput.value = "";
      if (pointsInput) pointsInput.value = "";
      if (timerDisplay) timerDisplay.textContent = "00:00 / 03:00";
      if (hudBadge) {
        hudBadge.textContent = "READY";
        hudBadge.style.background = "";
      }
      const badge = document.getElementById("speech-session-state-badge");
      if (badge) badge.textContent = "State: SETUP";
      const feedback = document.getElementById("speech-feedback-container");
      if (feedback) feedback.style.display = "none";
      updateSpeechPointsPreview();
      showToast("Stage cleared for a new rehearsal!");
    });
  }
}

async function analyzeSpeechRehearsal(transcript) {
  const container = document.getElementById("speech-feedback-container");
  const hudBadge = document.getElementById("speech-hud-state");
  const hintEl = document.getElementById("speech-status-hint");

  try {
    const res = await fetch("/api/v1/speech/analyze-transcript", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: currentStudentId,
        transcript: transcript,
        topic: speechTopic || "Presentation",
        required_points: speechRequiredPoints
      })
    });

    if (!res.ok) throw new Error("Analysis failed");
    const data = await res.json();
    const fb = data.feedback || {};

    if (hudBadge) {
      hudBadge.textContent = "COMPLETED";
      hudBadge.style.background = "#10b981";
    }
    if (hintEl) hintEl.textContent = "Rehearsal feedback generated below!";

    if (container) {
      container.style.display = "block";
      const overall = fb.overall_score || fb.score || 85;
      const missed = fb.missed_points || [];

      container.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:12px; margin-bottom:14px;">
          <h4 style="margin:0;">🎤 Rehearsal Rubric Score</h4>
          <span class="tag tag-accent" style="font-size:1.1rem; padding:4px 12px;">${overall} / 100</span>
        </div>
        <div class="grid-2-col" style="gap:14px; margin-bottom:14px;">
          <div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">Structure & Delivery</div>
            <strong style="color:#34d399;">${fb.structure_score || 80}/100</strong>
          </div>
          <div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:4px;">Grammar & Vocabulary</div>
            <strong style="color:#818cf8;">${fb.grammar_score || 85}/100</strong>
          </div>
        </div>
        ${missed.length > 0 ? `
          <div style="margin-bottom:12px;">
            <div style="color:#f43f5e; font-size:0.82rem; font-weight:600; margin-bottom:4px;">⚠️ Missing Points to Rehearse:</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
              ${missed.map(m => `<span class="tag tag-danger" style="font-size:0.75rem;">${escapeHtml(m)}</span>`).join("")}
            </div>
          </div>
        ` : `
          <div style="color:#34d399; font-size:0.82rem; margin-bottom:12px;">✓ All key presentation checkpoints covered!</div>
        `}
        <div style="font-size:0.85rem; line-height:1.5; color:var(--text-light); background:rgba(255,255,255,0.02); padding:10px; border-radius:6px;">
          <strong>Coach Summary:</strong> ${escapeHtml(fb.coach_summary || fb.summary || "Good delivery and pacing. Keep practicing to refine transitions.")}
        </div>
      `;
    }
  } catch (err) {
    console.error("Speech analysis error:", err);
    showToast("Error generating speech feedback", true);
  }
}

