/**
 * Mizo 3.0 — Virtual Robot Face Display Controller
 * ==================================================
 * Modular face display engine designed for:
 * 1. Web dashboard virtual robot face display.
 * 2. Future adaptation to physical OLED (SSD1306 128x64) and TFT (ST7789 240x240/320x240) displays.
 * 
 * Supports 5 core emotional/operational states:
 * - 'idle': Relaxed eyes, occasional natural blinking, subtle breathing movement.
 * - 'listening': Attentive eyes, vibrant pulsing listening halo/indicator.
 * - 'thinking': Cognitive processing with scanning eyes & animated thinking dots.
 * - 'speaking': Dynamic audio waveform mouth movement while Mizo speaks.
 * - 'error': Friendly, puzzled/concerned asymmetric eye expression and soft warning glow.
 * (Also supports 'onboarding' warm welcoming smile for backward compatibility).
 * 
 * Includes Developer Preview Mode:
 * - Allows testing all 5 states manually.
 * - Does NOT affect the ESP32 or backend operating state.
 * - Safely exits preview mode and restores live server-reported voice state.
 */

(function (window, document) {
  'use strict';

  // State metadata: labels, hints, and OLED/TFT geometric primitive descriptions
  const FACE_STATE_CONFIG = {
    idle: {
      label: 'Ready — Say "Hey Mizo" or tap to speak',
      hint: "Standing by • Say 'Hey Mizo' or tap to speak",
      dotClass: 'dot-idle',
      accentColor: '#38bdf8',
      hardware: {
        eyeShape: 'rounded_rect',
        eyeWidth: 16,
        eyeHeight: 22,
        pupilSize: 6,
        mouthType: 'line',
        mouthWidth: 20,
        listeningIndicator: false,
        thinkingMatrix: false,
        speakingWaves: false,
        colorRgb: [56, 189, 248]
      }
    },
    listening: {
      label: 'Listening...',
      hint: 'Microphone active • Capturing speech',
      dotClass: 'dot-listening',
      accentColor: '#22d3ee',
      hardware: {
        eyeShape: 'wide_capsule',
        eyeWidth: 18,
        eyeHeight: 24,
        pupilSize: 8,
        mouthType: 'dot',
        mouthWidth: 8,
        listeningIndicator: true,
        thinkingMatrix: false,
        speakingWaves: false,
        colorRgb: [34, 211, 238]
      }
    },
    thinking: {
      label: 'Thinking... Let me figure that out',
      hint: 'Processing speech & formulating response...',
      dotClass: 'dot-thinking',
      accentColor: '#c084fc',
      hardware: {
        eyeShape: 'scanning_capsule',
        eyeWidth: 16,
        eyeHeight: 20,
        pupilSize: 6,
        mouthType: 'short_line',
        mouthWidth: 14,
        listeningIndicator: false,
        thinkingMatrix: true,
        speakingWaves: false,
        colorRgb: [192, 132, 252]
      }
    },
    speaking: {
      label: 'Mizo is speaking...',
      hint: 'Delivering spoken response via speaker',
      dotClass: 'dot-speaking',
      accentColor: '#34d399',
      hardware: {
        eyeShape: 'expressive_capsule',
        eyeWidth: 16,
        eyeHeight: 22,
        pupilSize: 6,
        mouthType: 'dynamic_waves',
        mouthWidth: 26,
        listeningIndicator: false,
        thinkingMatrix: false,
        speakingWaves: true,
        colorRgb: [52, 211, 153]
      }
    },
    error: {
      label: 'Something went wrong. Please try again.',
      hint: "Voice input unclear or error • Please try again or tap to speak",
      dotClass: 'dot-error',
      accentColor: '#f43f5e',
      hardware: {
        eyeShape: 'tilted_capsules',
        eyeWidth: 16,
        eyeHeight: 20,
        pupilSize: 5,
        mouthType: 'wavy_line',
        mouthWidth: 18,
        listeningIndicator: false,
        thinkingMatrix: false,
        speakingWaves: false,
        colorRgb: [244, 63, 94]
      }
    },
    onboarding: {
      label: "Welcome! Let's get to know you",
      hint: 'Conducting personalized student interview',
      dotClass: 'dot-onboarding',
      accentColor: '#f59e0b',
      hardware: {
        eyeShape: 'arched_happy',
        eyeWidth: 18,
        eyeHeight: 12,
        pupilSize: 0,
        mouthType: 'curved_smile',
        mouthWidth: 24,
        listeningIndicator: false,
        thinkingMatrix: false,
        speakingWaves: false,
        colorRgb: [245, 158, 11]
      }
    }
  };

  class MizoFaceController {
    constructor() {
      this.currentState = 'idle';
      this.isPreviewMode = false;
      this.previewState = null;
      this.lastLiveState = 'idle';
      this.lastLiveLabel = null;
      this.lastLiveHint = null;

      this.panelEl = null;
      this.labelEl = null;
      this.hintEl = null;
      this.dotEl = null;
      this.modeBadgeEl = null;
      this.previewButtons = [];
      this.exitPreviewBtn = null;

      this.blinkTimer = null;
      this.blinkTimeoutId = null;
      this.gazeTimer = null;
      this.gazeResetTimer = null;
      this.reducedMotion = false;

      this._init();
    }

    _init() {
      // Check reduced-motion preference
      if (window.matchMedia) {
        const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
        this.reducedMotion = mediaQuery.matches;
        mediaQuery.addEventListener('change', (e) => {
          this.reducedMotion = e.matches;
          if (this.reducedMotion) {
            this._stopBlinking();
            this._stopGazeTracking();
          } else if (this.currentState === 'idle') {
            this._startBlinking();
            this._startGazeTracking();
          }
        });
      }

      // Initialize DOM elements when ready
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.bindDom());
      } else {
        this.bindDom();
      }
    }

    bindDom() {
      this.panelEl = document.getElementById('mizo-face-panel');
      this.labelEl = document.getElementById('mizo-face-status-label');
      this.hintEl = document.getElementById('mizo-face-status-hint');
      this.dotEl = document.getElementById('mizo-face-status-dot');
      this.modeBadgeEl = document.getElementById('mizo-face-mode-badge');
      this.exitPreviewBtn = document.getElementById('btn-preview-exit');

      // Bind preview buttons
      const btns = document.querySelectorAll('.btn-face-preview[data-preview]');
      this.previewButtons = Array.from(btns);

      this.previewButtons.forEach((btn) => {
        btn.addEventListener('click', (e) => {
          e.preventDefault();
          const targetState = btn.getAttribute('data-preview');
          this.setPreviewState(targetState);
        });
      });

      if (this.exitPreviewBtn) {
        this.exitPreviewBtn.addEventListener('click', (e) => {
          e.preventDefault();
          this.exitPreview();
        });
      }

      // Initial render
      this.render();
      if (!this.reducedMotion) {
        this._startBlinking();
        this._startGazeTracking();
      }
    }

    /**
     * Updates the face display.
     * If preview mode is active, live updates are cached without overriding the preview display.
     *
     * @param {string} state - 'idle' | 'listening' | 'thinking' | 'speaking' | 'error' | 'onboarding'
     * @param {string|null} labelOverride - Optional custom label
     * @param {string|null} hintOverride - Optional custom hint
     * @param {boolean} isLiveUpdate - Whether this update originates from live voice state/server
     */
    update(state, labelOverride = null, hintOverride = null, isLiveUpdate = true) {
      const validState = FACE_STATE_CONFIG[state] ? state : 'idle';

      if (this.isPreviewMode && isLiveUpdate) {
        // Cache the live state so exiting preview mode instantly restores it
        this.lastLiveState = validState;
        this.lastLiveLabel = labelOverride;
        this.lastLiveHint = hintOverride;
        return;
      }

      this.currentState = validState;
      if (isLiveUpdate) {
        this.lastLiveState = validState;
        this.lastLiveLabel = labelOverride;
        this.lastLiveHint = hintOverride;
      }

      this.render(labelOverride, hintOverride);
    }

    /**
     * Activates manual Developer Preview mode.
     * This switches the face visually WITHOUT sending any commands to the ESP32 or backend.
     */
    setPreviewState(state) {
      if (!FACE_STATE_CONFIG[state]) return;
      this.isPreviewMode = true;
      this.previewState = state;
      this.currentState = state;

      this.updatePreviewUi(state);
      this.render();
    }

    /**
     * Exits Developer Preview mode and restores real-time server voice state.
     */
    exitPreview() {
      this.isPreviewMode = false;
      this.previewState = null;
      this.currentState = this.lastLiveState || 'idle';

      this.updatePreviewUi(null);
      this.render(this.lastLiveLabel, this.lastLiveHint);

      // Trigger immediate live check if available in window context
      if (typeof window.checkESP32HardwareStatus === 'function') {
        window.checkESP32HardwareStatus();
      }
    }

    updatePreviewUi(activePreviewState) {
      // Update Mode Badge
      if (this.modeBadgeEl) {
        if (this.isPreviewMode && activePreviewState) {
          this.modeBadgeEl.className = 'mizo-mode-pill preview';
          this.modeBadgeEl.textContent = `PREVIEW: ${activePreviewState.toUpperCase()}`;
        } else {
          this.modeBadgeEl.className = 'mizo-mode-pill live';
          this.modeBadgeEl.textContent = '● LIVE MODE';
        }
      }

      // Update button active states
      this.previewButtons.forEach((btn) => {
        const btnState = btn.getAttribute('data-preview');
        if (this.isPreviewMode && btnState === activePreviewState) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });

      if (this.exitPreviewBtn) {
        if (this.isPreviewMode) {
          this.exitPreviewBtn.classList.add('visible');
        } else {
          this.exitPreviewBtn.classList.remove('visible');
        }
      }
    }

    render(labelOverride = null, hintOverride = null) {
      if (!this.panelEl) {
        this.panelEl = document.getElementById('mizo-face-panel');
        if (!this.panelEl) return;
      }

      const cfg = FACE_STATE_CONFIG[this.currentState] || FACE_STATE_CONFIG.idle;

      // Update data-state attribute
      this.panelEl.setAttribute('data-state', this.currentState);

      // Update Labels
      if (!this.labelEl) this.labelEl = document.getElementById('mizo-face-status-label');
      if (!this.hintEl) this.hintEl = document.getElementById('mizo-face-status-hint');
      if (!this.dotEl) this.dotEl = document.getElementById('mizo-face-status-dot');

      if (this.labelEl) {
        this.labelEl.textContent = labelOverride || cfg.label;
      }

      if (this.hintEl) {
        if (this.isPreviewMode) {
          this.hintEl.textContent = '(Preview mode active • Live updates paused)';
        } else {
          this.hintEl.textContent = hintOverride || cfg.hint;
        }
      }

      // Update Status Indicator Dot & Icons
      if (this.dotEl) {
        this.dotEl.className = `mizo-status-indicator-dot ${cfg.dotClass}`;
        if (this.currentState === 'thinking') {
          this.dotEl.innerHTML = '<span class="status-mini-spinner" aria-hidden="true"></span>';
        } else if (this.currentState === 'speaking') {
          this.dotEl.innerHTML = '<span class="status-mini-wave" aria-hidden="true"><i></i><i></i><i></i></span>';
        } else {
          this.dotEl.innerHTML = '';
        }
      }

      // Handle Animation Lifecycles by State
      if (this.reducedMotion) {
        this._stopBlinking();
        this._stopGazeTracking();
        return;
      }

      if (this.currentState === 'idle') {
        this._startBlinking();
        this._startGazeTracking();
      } else if (this.currentState === 'listening') {
        this._stopBlinking();
        this._stopGazeTracking();
        this._applyGaze('gaze-center');
      } else if (this.currentState === 'thinking') {
        this._stopBlinking();
        this._stopGazeTracking();
        this._applyGaze('gaze-thinking');
      } else if (this.currentState === 'speaking') {
        this._stopGazeTracking();
        this._applyGaze('gaze-center');
        this._startBlinking(); // Occasional natural blink during speech
      } else if (this.currentState === 'error') {
        this._stopBlinking();
        this._stopGazeTracking();
        this._applyGaze('gaze-puzzled');
      } else {
        this._startBlinking();
        this._stopGazeTracking();
      }
    }

    /**
     * Natural intermittent blinking:
     * Blinks every 2–5 seconds with randomized intervals and occasional double blinks.
     */
    _startBlinking() {
      if (this.blinkTimer || this.reducedMotion) return;
      const scheduleBlink = () => {
        // Natural intermittent blinking between 2.0s and 5.0s
        const nextBlinkMs = 2000 + Math.random() * 3000;
        this.blinkTimer = setTimeout(() => {
          if ((this.currentState === 'idle' || this.currentState === 'speaking' || this.currentState === 'onboarding') && this.panelEl) {
            const isDouble = Math.random() < 0.25; // 25% chance of double blink
            this._performBlink(isDouble, () => {
              if (this.currentState === 'idle' || this.currentState === 'speaking' || this.currentState === 'onboarding') {
                scheduleBlink();
              } else {
                this._stopBlinking();
              }
            });
          } else {
            this._stopBlinking();
          }
        }, nextBlinkMs);
      };
      scheduleBlink();
    }

    _performBlink(isDouble, callback) {
      if (!this.panelEl) {
        if (callback) callback();
        return;
      }
      this.panelEl.classList.add('mizo-blink-now');
      this.blinkTimeoutId = setTimeout(() => {
        if (this.panelEl) this.panelEl.classList.remove('mizo-blink-now');
        if (isDouble) {
          this.blinkTimeoutId = setTimeout(() => {
            if (this.panelEl) this.panelEl.classList.add('mizo-blink-now');
            this.blinkTimeoutId = setTimeout(() => {
              if (this.panelEl) this.panelEl.classList.remove('mizo-blink-now');
              if (callback) callback();
            }, 100);
          }, 80);
        } else {
          if (callback) callback();
        }
      }, 130);
    }

    _stopBlinking() {
      if (this.blinkTimer) {
        clearTimeout(this.blinkTimer);
        this.blinkTimer = null;
      }
      if (this.blinkTimeoutId) {
        clearTimeout(this.blinkTimeoutId);
        this.blinkTimeoutId = null;
      }
      if (this.panelEl) {
        this.panelEl.classList.remove('mizo-blink-now');
      }
    }

    /**
     * Natural Gaze Tracking:
     * Eyes subtly look left, right, up, down, then return to center.
     */
    _startGazeTracking() {
      if (this.gazeTimer || this.reducedMotion) return;
      const directions = ['gaze-left', 'gaze-right', 'gaze-up', 'gaze-down'];
      const scheduleGaze = () => {
        const nextGazeMs = 2600 + Math.random() * 3000;
        this.gazeTimer = setTimeout(() => {
          if (this.currentState === 'idle' && this.panelEl) {
            if (Math.random() < 0.7) {
              const dir = directions[Math.floor(Math.random() * directions.length)];
              this._applyGaze(dir);
              const glanceMs = 1100 + Math.random() * 700;
              this.gazeResetTimer = setTimeout(() => {
                this._applyGaze('gaze-center');
                scheduleGaze();
              }, glanceMs);
            } else {
              this._applyGaze('gaze-center');
              scheduleGaze();
            }
          } else {
            this._stopGazeTracking();
          }
        }, nextGazeMs);
      };
      scheduleGaze();
    }

    _applyGaze(directionClass) {
      if (!this.panelEl) return;
      const gazeClasses = ['gaze-left', 'gaze-right', 'gaze-up', 'gaze-down', 'gaze-center', 'gaze-thinking', 'gaze-puzzled'];
      this.panelEl.classList.remove(...gazeClasses);
      if (directionClass) {
        this.panelEl.classList.add(directionClass);
      }
    }

    _stopGazeTracking() {
      if (this.gazeTimer) {
        clearTimeout(this.gazeTimer);
        this.gazeTimer = null;
      }
      if (this.gazeResetTimer) {
        clearTimeout(this.gazeResetTimer);
        this.gazeResetTimer = null;
      }
      this._applyGaze('gaze-center');
    }

    /**
     * Hardware Display Export Interface.
     * Returns a JSON-serializable geometric frame state suitable for direct C++/Arduino
     * rendering on a small SSD1306 OLED (128x64) or ST7789 TFT (240x240) screen.
     */
    getDisplayFrame(targetWidth = 128, targetHeight = 64) {
      const cfg = FACE_STATE_CONFIG[this.currentState] || FACE_STATE_CONFIG.idle;
      const hw = cfg.hardware;
      const scaleX = targetWidth / 128;
      const scaleY = targetHeight / 64;

      return {
        state: this.currentState,
        isPreview: this.isPreviewMode,
        resolution: { width: targetWidth, height: targetHeight },
        accentColor: cfg.accentColor,
        colorRgb: hw.colorRgb,
        leftEye: {
          shape: hw.eyeShape,
          x: Math.round(36 * scaleX),
          y: Math.round(24 * scaleY),
          width: Math.round(hw.eyeWidth * scaleX),
          height: Math.round(hw.eyeHeight * scaleY),
          pupilRadius: Math.round(hw.pupilSize * scaleX)
        },
        rightEye: {
          shape: hw.eyeShape,
          x: Math.round((128 - 36 - hw.eyeWidth) * scaleX),
          y: Math.round(24 * scaleY),
          width: Math.round(hw.eyeWidth * scaleX),
          height: Math.round(hw.eyeHeight * scaleY),
          pupilRadius: Math.round(hw.pupilSize * scaleX)
        },
        mouth: {
          type: hw.mouthType,
          x: Math.round((64 - hw.mouthWidth / 2) * scaleX),
          y: Math.round(50 * scaleY),
          width: Math.round(hw.mouthWidth * scaleX)
        },
        listeningIndicator: hw.listeningIndicator,
        thinkingMatrix: hw.thinkingMatrix,
        speakingWaves: hw.speakingWaves
      };
    }
  }

  // Instantiate singleton
  const mizoFaceInstance = new MizoFaceController();

  // Expose to window
  window.MizoFaceController = MizoFaceController;
  window.mizoFace = mizoFaceInstance;
  window.FACE_STATE_CONFIG = FACE_STATE_CONFIG;

})(window, document);
