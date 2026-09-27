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
  STANDBY: "STANDBY",
  WAKE_ONLY: "WAKE_ONLY",
  ACTIVE_LISTENING: "ACTIVE_LISTENING",
  CAPTURING: "CAPTURING",
  PROCESSING: "PROCESSING",
  SPEAKING: "SPEAKING",
  STOPPING: "STOPPING",
  ASSESSMENT_RECORDING: "ASSESSMENT_RECORDING"
};

let currentVoiceState = VoiceState.STANDBY;
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
    "tab-simulator": "Onboarding & Voice Chat",
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
    if (statMod) statMod.textContent = data[`${activeProvider}_model`] || "Llama 3.3 70B";
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
    if (groqMod) groqMod.value = data.groq_model || "llama-3.3-70b-versatile";
    const openaiMod = document.getElementById("input-openai-model");
    if (openaiMod) openaiMod.value = data.openai_model || "gpt-4o-mini";
    const ollamaUrl = document.getElementById("input-ollama-url");
    if (ollamaUrl) ollamaUrl.value = data.ollama_base_url || "http://localhost:11434";

    const groqKeyStat = document.getElementById("groq-key-status");
    if (groqKeyStat) groqKeyStat.textContent = `Status: ${data.groq_api_key_masked || "Not set"}`;
    const openaiKeyStat = document.getElementById("openai-key-status");
    if (openaiKeyStat) openaiKeyStat.textContent = `Status: ${data.openai_api_key_masked || "Not set"}`;
  } catch (err) {
    console.error("Error loading settings:", err);
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
      groq: "Groq (Llama 3.3 70B)",
      openai: "OpenAI (GPT-4o)",
      qwen: "Qwen 2.5 (OpenRouter)",
      ollama: "Local Ollama"
    };

    container.innerHTML = data.providers.map(p => {
      const isConnected = p.status === "connected";
      const isConfigured = p.configured;
      const statusClass = isConnected ? "badge-success" : (isConfigured ? "badge-danger" : "badge-neutral");
      const statusText = isConnected ? `Connected (${p.latency_ms}ms)` : (isConfigured ? "Unavailable" : "Not configured");

      return `
        <div class="provider-status-card" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong>${providerNames[p.provider] || p.provider}</strong>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Model: ${escapeHtml(p.model)}</div>
          </div>
          <span class="badge ${statusClass}">${statusText}</span>
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
    updateSessionBar("session-fluency", session.metrics.fluency_score);
    updateSessionBar("session-grammar", session.metrics.grammar_accuracy);
    updateSessionBar("session-vocab", session.metrics.vocabulary_richness);
    updateSessionBar("session-pacing", session.metrics.pacing_score);
  } catch (err) {
    console.error("Error loading active session:", err);
  }
}

function updateSessionBar(idPrefix, val) {
  val = Math.round(val || 0);
  const textEl = document.getElementById(`${idPrefix}-text`);
  const barEl = document.getElementById(`${idPrefix}-bar`);
  if (textEl) textEl.textContent = `${val}.0%`;
  if (barEl) barEl.style.width = `${val}%`;
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

      updateBar("fluency", s.fluency_score);
      updateBar("grammar", s.grammar_score);
      updateBar("vocab", s.vocabulary_score);
      updateBar("conf", s.confidence_score);

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

function updateBar(key, val) {
  val = Math.round(val || 75);
  const textEl = document.getElementById(`snap-${key}`);
  const barEl = document.getElementById(`bar-${key}`);
  if (textEl) textEl.textContent = `${val}.0%`;
  if (barEl) barEl.style.width = `${val}%`;
}

// -------------------------------------------------------------
// Conversation Stream & History
// -------------------------------------------------------------
async function loadConversations(searchQuery = null) {
  try {
    let url = `/api/v1/conversations?limit=50`;
    if (searchQuery && searchQuery.trim()) {
      url += `&search=${encodeURIComponent(searchQuery.trim())}`;
    }

    const res = await fetch(url);
    if (!res.ok) return;
    const logs = await res.json();

    // 1. Render in Live Chat Box on Onboarding Studio
    const chatBox = document.getElementById("chat-messages-container");
    if (chatBox) {
      if (logs.length === 0) {
        chatBox.innerHTML = `
          <div class="chat-bubble assistant">
            <div class="chat-avatar">🤖</div>
            <div class="chat-body">
              <div class="chat-meta"><strong>Mizo AI</strong> • Just now</div>
              <div class="chat-text">Hello! I'm Mizo, your AI English learning robot tutor. To get started, what should I call you?</div>
            </div>
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

  if (newState === VoiceState.STANDBY || newState === VoiceState.WAKE_ONLY) {
    console.log("[VOICE] State: SLEEP");
    console.log("[VOICE] Wake listener: ACTIVE");
    console.log("[VOICE] Microphone: ACTIVE");
  } else if (newState === VoiceState.ACTIVE_LISTENING) {
    console.log("[VOICE] State: LISTENING");
  }

  const stateBadge = document.getElementById("voice-state-badge");
  if (stateBadge) {
    if (isMicMuted) {
      stateBadge.className = "voice-state-badge muted";
      stateBadge.textContent = "MIC MUTED";
    } else {
      const stateMap = {
        [VoiceState.STANDBY]: { cls: "wake-only", label: "SLEEP • Say 'Mizo'" },
        [VoiceState.WAKE_ONLY]: { cls: "wake-only", label: "SLEEP • Say 'Mizo'" },
        [VoiceState.ACTIVE_LISTENING]: { cls: "active-listening", label: "ACTIVE_LISTENING" },
        [VoiceState.CAPTURING]: { cls: "capturing", label: "CAPTURING" },
        [VoiceState.PROCESSING]: { cls: "processing", label: "PROCESSING" },
        [VoiceState.SPEAKING]: { cls: "speaking", label: "SPEAKING" },
        [VoiceState.STOPPING]: { cls: "stopping", label: "STOPPING" },
        [VoiceState.ASSESSMENT_RECORDING]: { cls: "assessment-recording", label: "ASSESSMENT_RECORDING" }
      };
      const info = stateMap[newState] || { cls: "wake-only", label: newState };
      stateBadge.className = `voice-state-badge ${info.cls}`;
      stateBadge.textContent = info.label;
    }
  }

  // State-specific lifecycle management
  if (newState === VoiceState.STANDBY || newState === VoiceState.WAKE_ONLY || newState === VoiceState.STOPPING) {
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

    setSilenceBadge(isMicMuted ? "Microphone Muted" : "💤 Sleep • Say 'Mizo'", "normal");

    // Keep local wake detector active in standby
    if (isWakeWordEnabled && !isMicMuted) {
      initWakeWordListener();
    }
  } else if (newState === VoiceState.ACTIVE_LISTENING) {
    // Stop local wake detector so it doesn't conflict with conversational audio
    if (wakeWordRecognizer) {
      try { wakeWordRecognizer.abort(); } catch(e) {}
    }
    hideAssessmentHUD();
    setSilenceBadge("🟢 Listening... (Speak naturally)", "playing");
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
  } else if (newState === VoiceState.ASSESSMENT_RECORDING) {
    clearTimeout(inactivityTimerId);
    silenceStartTimestamp = null;
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
      setVoiceState(VoiceState.STANDBY, "inactivity_snooze");
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

function handleWakeWordDetected(fullTranscript) {
  if (isMicMuted || (currentVoiceState !== VoiceState.WAKE_ONLY && currentVoiceState !== VoiceState.STANDBY)) return;

  console.log(`[VOICE] Wake word detected: Mizo`);
  console.log(`[VOICE] Waking assistant`);

  const remainingText = extractMizoTrailingText(fullTranscript);

  showToast("🎙️ 'Mizo' recognized! Active mode enabled.");

  if (remainingText.length > 0) {
    // Treat trailing words as direct user input
    setVoiceState(VoiceState.ACTIVE_LISTENING, "wake_with_speech");
    const textInput = document.getElementById("sim-text-input");
    if (textInput) textInput.value = remainingText;
    sendTextMessage(remainingText);
  } else {
    // Wake phrase alone -> transition to ACTIVE_LISTENING & listen for speech immediately
    setVoiceState(VoiceState.ACTIVE_LISTENING, "wake_phrase");
    const chatBox = document.getElementById("chat-messages-container");
    const hasHistory = chatBox && chatBox.querySelectorAll(".chat-bubble").length > 1;
    if (!hasHistory) {
      triggerMizoStartGreeting();
    } else {
      sendTextMessage("Mizo");
    }
  }
}

function executeMizoStop() {
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

  // 4. Return to STANDBY mode
  setVoiceState(VoiceState.STANDBY, "mizo_stop_command");
  showToast("🛑 'Mizo stop' executed. Standby mode active.");
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

      // 2. Dedicated Assessment Recording: normal 3s turn silence timer is DISABLED!
      if (currentVoiceState === VoiceState.ASSESSMENT_RECORDING) {
        if (rms > currentSpeechThreshold) {
          setSilenceBadge("🎙️ Recording Assessment (Speak continuously)...", "speaking");
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

  assessmentStartTime = Date.now();
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
      finishSpeakingAssessment();
    }
  }, 1000);
}

function hideAssessmentHUD() {
  const hud = document.getElementById("assessment-hud-box");
  if (hud) hud.style.display = "none";
  clearInterval(assessmentTimerInterval);
}

function finishSpeakingAssessment() {
  const elapsedSecs = assessmentStartTime ? Math.round((Date.now() - assessmentStartTime) / 1000) : 0;
  hideAssessmentHUD();

  if (elapsedSecs < 20) {
    showToast(`Speech sample was only ${elapsedSecs}s (too short). Please speak for about 1 minute on the topic.`, true);
    const modal = document.getElementById("modal-short-recording");
    if (modal) modal.style.display = "flex";
    return;
  }

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
      if (!isWakeWordEnabled || isMicMuted || (currentVoiceState !== VoiceState.WAKE_ONLY && currentVoiceState !== VoiceState.STANDBY)) return;

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
        console.log("[VOICE] Wake word not detected. Ignoring.");
      }
    };

    wakeWordRecognizer.onerror = (e) => {
      if (e.error !== "no-speech" && e.error !== "aborted") {
        console.log("[Wake Word Engine]", e.error);
      }
      if (isWakeWordEnabled && !isMicMuted && (currentVoiceState === VoiceState.WAKE_ONLY || currentVoiceState === VoiceState.STANDBY)) {
        setTimeout(() => {
          try { wakeWordRecognizer.start(); } catch (err) {}
        }, 800);
      }
    };

    wakeWordRecognizer.onend = () => {
      if (isWakeWordEnabled && !isMicMuted && (currentVoiceState === VoiceState.WAKE_ONLY || currentVoiceState === VoiceState.STANDBY)) {
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

    if (currentVoiceState === VoiceState.WAKE_ONLY) {
      console.log("[Voice SM] Late response discarded due to stop command.");
      return;
    }

    if (data.is_control_command && data.command === "stop") {
      executeMikazaStop();
      return;
    }

    if (recordStatus) recordStatus.textContent = "Ready";
    if (timerEl) timerEl.textContent = "00:00 / 01:00";

    handleInteractionResponse(data);
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("[Voice SM] Audio upload aborted by stop command.");
      return;
    }
    currentAbortController = null;
    if (recordStatus) recordStatus.textContent = "Error processing audio";
    setSilenceBadge("STT Error", "normal");
    console.error(err);

    if (currentVoiceState !== VoiceState.WAKE_ONLY) {
      setTimeout(() => setVoiceState(VoiceState.ACTIVE_LISTENING, "error_resume"), 1200);
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
  setVoiceState(VoiceState.PROCESSING, "sending_text");

  try {
    const res = await fetch("/api/v1/esp32/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        student_id: currentStudentId,
        session_id: activeSessionId,
        mode: "coach"
      }),
      signal: currentAbortController.signal
    });
    const data = await res.json();
    currentAbortController = null;

    if (currentVoiceState === VoiceState.WAKE_ONLY) {
      console.log("[Voice SM] Late response discarded due to stop command.");
      return;
    }

    handleInteractionResponse(data);
  } catch (err) {
    if (err.name === "AbortError") {
      console.log("[Voice SM] Text request aborted by stop command.");
      return;
    }
    currentAbortController = null;
    console.error("Chat error:", err);
    if (currentVoiceState !== VoiceState.WAKE_ONLY) {
      setVoiceState(VoiceState.ACTIVE_LISTENING, "chat_error_fallback");
    }
  }
}

function handleInteractionResponse(data) {
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
    // If no audio was returned, resume listening if in voice mode or transition to assessment
    if (data.onboarding_step === "speech_test_prompt" || data.onboarding_step === "speaking_assessment") {
      setVoiceState(VoiceState.ASSESSMENT_RECORDING, "assessment_prompt_no_audio");
      showAssessmentHUD(data.topic || "1-Minute Speaking Sample");
      if (!isRecording && !isMicMuted) {
        setTimeout(() => startVoiceRecording(true), 500);
      }
    } else if (!isRecording && !isMicMuted) {
      setVoiceState(VoiceState.ACTIVE_LISTENING, "no_audio_auto_listen");
      setTimeout(() => startVoiceRecording(true), 600);
    }
  }

  // 2. Update Session Metrics & Cumulative Progress
  if (data.session_metrics) {
    updateSessionBar("session-fluency", data.session_metrics.fluency_score);
    updateSessionBar("session-grammar", data.session_metrics.grammar_accuracy);
    updateSessionBar("session-vocab", data.session_metrics.vocabulary_richness);
    updateSessionBar("session-pacing", data.session_metrics.pacing_score);
  }

  if (data.historical_metrics) {
    updateBar("grammar", data.historical_metrics.grammar_score);
    updateBar("vocab", data.historical_metrics.vocabulary_score);
    updateBar("fluency", data.historical_metrics.fluency_score);
    updateBar("conf", data.historical_metrics.confidence_score);
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
}

function renderLatestAssessmentPanel(a) {
  const panel = document.getElementById("latest-assessment-panel");
  const header = document.getElementById("assessment-summary-header");
  const content = document.getElementById("latest-assessment-content");
  if (!panel || !content) return;

  panel.style.display = "block";
  if (header) {
    header.textContent = `Baseline Assessment Completed: ${a.overall_level || 'Intermediate'} (${a.communication_score || 75}% Communication Score)`;
  }

  content.innerHTML = `
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px;">
      <div><span class="text-muted">Grammar:</span> <strong>${a.grammar_score}%</strong></div>
      <div><span class="text-muted">Vocabulary:</span> <strong>${a.vocabulary_score}%</strong></div>
      <div><span class="text-muted">Fluency:</span> <strong>${a.fluency_score}%</strong></div>
      <div><span class="text-muted">Pronunciation:</span> <strong>${a.pronunciation_score}%</strong></div>
      <div><span class="text-muted">Confidence:</span> <strong>${a.confidence_score}%</strong></div>
      <div><span class="text-muted">Overall:</span> <strong>${a.overall_level}</strong></div>
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

  // 7. Provider Card Selection
  document.querySelectorAll(".provider-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".provider-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      activeProvider = card.getAttribute("data-provider");
    });
  });

  // 8. Toggle Password Visibility
  document.querySelectorAll(".btn-toggle-key").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const input = document.getElementById(targetId);
      input.type = input.type === "password" ? "text" : "password";
      btn.textContent = input.type === "password" ? "👁️" : "🙈";
    });
  });

  // 9. Save AI Behavior
  const btnSaveBehavior = document.getElementById("btn-save-behavior");
  if (btnSaveBehavior) {
    btnSaveBehavior.addEventListener("click", async () => {
      const payload = {
        system_prompt: document.getElementById("setting-system-prompt").value,
        coaching_mode: document.getElementById("setting-coaching-mode").value,
        grammar_strictness: document.getElementById("setting-grammar-strictness").value,
        tts_voice: document.getElementById("setting-tts-voice").value,
        tts_rate: document.getElementById("setting-tts-rate").value,
      };
      await saveSettingsAPI(payload);
    });
  }

  // 10. Save AI Providers & Keys
  const btnSaveProviders = document.getElementById("btn-save-providers");
  if (btnSaveProviders) {
    btnSaveProviders.addEventListener("click", async () => {
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
      await loadProviderStatus();
    });
  }

  // 11. Add Device
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
      showToast("Settings & custom prompt saved successfully!");
      await loadSettings();
      await loadProviderStatus();
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
      const groqModel = document.getElementById("input-groq-model")?.value || "llama-3.3-70b-versatile";
      const groqKey = document.getElementById("input-groq-key")?.value || "";
      const openaiModel = document.getElementById("input-openai-model")?.value || "gpt-4o-mini";
      const openaiKey = document.getElementById("input-openai-key")?.value || "";
      const ollamaUrl = document.getElementById("input-ollama-url")?.value || "http://localhost:11434";

      const payload = {
        active_provider: activeProvider,
        groq_model: groqModel,
        openai_model: openaiModel,
        ollama_base_url: ollamaUrl
      };

      if (groqKey.trim()) payload.groq_api_key = groqKey.trim();
      if (openaiKey.trim()) payload.openai_api_key = openaiKey.trim();

      await saveSettingsAPI(payload);
    });
  }

  // Provider Card Selection
  document.querySelectorAll(".provider-card").forEach(card => {
    card.addEventListener("click", () => {
      const prov = card.getAttribute("data-provider");
      if (prov) {
        activeProvider = prov;
        document.querySelectorAll(".provider-card").forEach(c => {
          c.classList.toggle("active", c.getAttribute("data-provider") === activeProvider);
        });
        showToast(`Selected primary provider: ${formatProviderName(activeProvider)}`);
      }
    });
  });

  // Test Provider Connections
  const btnTestProviders = document.getElementById("btn-test-providers");
  if (btnTestProviders) {
    btnTestProviders.addEventListener("click", async () => {
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
