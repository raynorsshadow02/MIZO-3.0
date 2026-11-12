/*
 * =================================================================================
 * MIZO 3.0 — ESP32-S3 ROBOTIC TUTOR FIRMWARE
 * Mode: WAKE-WORD STANDBY & CONTINUOUS CONVERSATION STATE MACHINE
 * Microcontroller: ESP32-S3 (WROOM-1 / N8R2 / N16R8)
 * Audio In:  INMP441 I2S MEMS Microphone (32-bit slot -> 16-bit 16kHz PCM WAV)
 * Audio Out: MAX98357A I2S Class-D DAC Amplifier + Speaker (16-bit 16kHz PCM)
 * =================================================================================
 *
 * HARDWARE WIRING (Verified Physical Hardware):
 * ---------------------------------------------------------------------------------
 * INMP441 (Microphone) -> ESP32-S3:
 *   VDD  --> 3.3V
 *   GND  --> GND
 *   L/R  --> GND (Left Channel)
 *   SCK  --> GPIO 4  (Serial Clock / BCLK)
 *   WS   --> GPIO 5  (Word Select / LRCLK)
 *   SD   --> GPIO 6  (Serial Data Output / DOUT)
 *
 * MAX98357A (Speaker DAC) -> ESP32-S3:
 *   VIN  --> 5V (recommended for loud 3.2W output; or 3.3V)
 *   GND  --> GND
 *   BCLK --> GPIO 18 (Bit Clock)
 *   LRC  --> GPIO 21 (Left/Right Clock / WS)
 *   DIN  --> GPIO 20 (Data In)
 *   GAIN --> Floating (Unconnected = 12dB gain) or GND (9dB gain)
 *
 * MANUAL CONTROLS:
 *   BOOT Button --> GPIO 0 (Press to manually wake or trigger turn)
 * =================================================================================
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <math.h>

// ---------------------------------------------------------------------------------
// 1. WI-FI & SERVER CREDENTIALS
// ---------------------------------------------------------------------------------
#if __has_include("secrets.h")
  #include "secrets.h"
  #ifndef SECRET_WIFI_SSID
    #define SECRET_WIFI_SSID     "SHAHID-4G"
  #endif
  #ifndef SECRET_WIFI_PASSWORD
    #define SECRET_WIFI_PASSWORD "kingshaikh"
  #endif
  const char* WIFI_SSID     = SECRET_WIFI_SSID;
  const char* WIFI_PASSWORD = SECRET_WIFI_PASSWORD;
#else
  const char* WIFI_SSID     = "SHAHID-4G";
  const char* WIFI_PASSWORD = "kingshaikh";
#endif

#ifdef SECRET_SERVER_IP
  const char* MIZO_SERVER_IP   = SECRET_SERVER_IP;
#else
  const char* MIZO_SERVER_IP   = "10.53.23.52";
#endif

#ifdef SECRET_SERVER_PORT
  const int   MIZO_SERVER_PORT = SECRET_SERVER_PORT;
#else
  const int   MIZO_SERVER_PORT = 8000;
#endif

const char* DEVICE_API_KEY     = "mizo_esp32_secret_key_123";

// ---------------------------------------------------------------------------------
// 2. AUDIO & VAD CONFIGURATION
// ---------------------------------------------------------------------------------
#define SAMPLE_RATE                 16000
#define MAX_RECORD_SECONDS          6
#define TOTAL_MAX_SAMPLES           (SAMPLE_RATE * MAX_RECORD_SECONDS)
#define BUFFER_SIZE_BYTES           (TOTAL_MAX_SAMPLES * sizeof(int16_t)) // 192,000 bytes

// Microphone Pins (INMP441)
#define I2S_MIC_PORT                I2S_NUM_0
#define I2S_MIC_SCK                 4
#define I2S_MIC_WS                  5
#define I2S_MIC_SD                  6

// Speaker DAC Pins (MAX98357A)
#define I2S_SPK_PORT                I2S_NUM_1
#define I2S_SPK_BCLK                18
#define I2S_SPK_LRC                 21
#define I2S_SPK_DIN                 20

// BOOT button on ESP32-S3
#define BUTTON_PIN                  0

// VAD & Timing Thresholds
#define VAD_CHUNK_SAMPLES           256     // 16ms chunk at 16kHz
#define SPEECH_ENERGY_THRESHOLD     1100    // Energy threshold on shifted 16-bit PCM
#define REQUIRED_SPEECH_HITS        2       // 32ms sustained energy to confirm speech start
#define SILENCE_TIMEOUT_MS          1200    // 1.2s silence confirms utterance completion
#define STANDBY_MAX_RECORD_MS       3500    // Max record window for wake-word attempt in STANDBY
#define ACTIVE_MAX_RECORD_MS        6000    // Max record window for dialogue turn in ACTIVE
#define ECHO_COOLDOWN_MS            750     // Cooldown after speaker playback to prevent echo
#define INACTIVITY_TIMEOUT_MS       35000   // 35s without speech in ACTIVE_LISTENING -> STANDBY

// Speaker Volume & Clarity
#define TEST_TONE_AMPLITUDE         28000   // 440Hz test sine tone (clean, punchy maximum volume)
#define PLAYBACK_GAIN               1.50f   // +3.5 dB boosted playback gain with smooth saturation clamp

// ---------------------------------------------------------------------------------
// 3. EXPLICIT ROBOT STATES
// ---------------------------------------------------------------------------------
enum RobotState {
    STATE_STANDBY,          // Waiting for "Hey Mizo" wake phrase
    STATE_ACTIVE_LISTENING, // Continuous conversation: waiting for user utterance
    STATE_RECORDING,        // User is speaking: recording with silence VAD
    STATE_PROCESSING,       // Audio sent to cloud server: waiting for transcription & response
    STATE_SPEAKING,         // Playing received WAV speech through MAX98357A speaker
    STATE_STOPPING          // Stop command received: halting conversation, returning to STANDBY
};

RobotState currentState = STATE_STANDBY;
const char* getStateName(RobotState state) {
    switch (state) {
        case STATE_STANDBY:          return "STANDBY";
        case STATE_ACTIVE_LISTENING: return "ACTIVE_LISTENING";
        case STATE_RECORDING:        return "RECORDING";
        case STATE_PROCESSING:       return "PROCESSING";
        case STATE_SPEAKING:         return "SPEAKING";
        case STATE_STOPPING:         return "STOPPING";
        default:                     return "UNKNOWN";
    }
}

void transitionTo(RobotState newState, const char* reason) {
    if (currentState == newState) return;
    Serial.printf("\n[STATE] %s -> %s (Reason: %s)\n", getStateName(currentState), getStateName(newState), reason);
    currentState = newState;
}

// Global Buffers & Guards
uint8_t* record_buffer = NULL;
unsigned long lastActiveTurnTime = 0;
unsigned long vadCooldownUntil = 0;
bool isProcessingTurn = false;

// ---------------------------------------------------------------------------------
// 4. AUDIO CONVERSION & CLIPPING PROTECTION
// ---------------------------------------------------------------------------------
// INMP441 sends 24-bit audio in upper bits of 32-bit slot. >> 14 gives clean 16-bit PCM.
inline int16_t convertSlot32ToPcm16(int32_t slot) {
    int32_t sample = slot >> 14;
    if (sample > 32767)  sample = 32767;
    if (sample < -32768) sample = -32768;
    return (int16_t)sample;
}

// Configurable playback gain with smooth soft-knee saturation protection (prevents flat-top distortion)
inline int16_t applyGainAndClamp(int16_t sample, float gain) {
    if (gain == 1.0f) return sample;
    int32_t amplified = (int32_t)(sample * gain);
    // Smooth asymptotic saturation above 28,000 to eliminate flat-top clipping distortion
    if (amplified > 28000) {
        int32_t excess = amplified - 28000;
        amplified = 28000 + (excess * 4767) / (excess + 4767);
    } else if (amplified < -28000) {
        int32_t excess = -amplified - 28000;
        amplified = -(28000 + (excess * 4768) / (excess + 4768));
    }
    if (amplified > 32767)  amplified = 32767;
    if (amplified < -32768) amplified = -32768;
    return (int16_t)amplified;
}

// ---------------------------------------------------------------------------------
// 5. I2S PERIPHERAL INITIALIZATION
// ---------------------------------------------------------------------------------
void setupI2SMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 512,
        .use_apll = false
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_SCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD
    };

    i2s_driver_install(I2S_MIC_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_MIC_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_MIC_PORT);
    Serial.println("✓ [INMP441] Microphone initialized (16kHz, 32-bit slot RX).");
}

void setupI2SSpeaker() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT, // Dual channel: drives both L & R to eliminate (L+R)/2 attenuation
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = true
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SPK_BCLK,
        .ws_io_num = I2S_SPK_LRC,
        .data_out_num = I2S_SPK_DIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    i2s_driver_install(I2S_SPK_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_SPK_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    Serial.println("✓ [MAX98357A] Speaker initialized (16kHz, 16-bit PCM TX Stereo Dual-Drive).");
}

// ---------------------------------------------------------------------------------
// 6. WAV HEADER GENERATION (16kHz, 16-bit Mono RIFF WAV)
// ---------------------------------------------------------------------------------
void setupWavHeader(byte* header, int totalDataLen) {
    long totalAudioLen = totalDataLen;
    long totalDataLenPlusHeader = totalAudioLen + 36;
    long longSampleRate = SAMPLE_RATE;
    int channels = 1;
    long byteRate = SAMPLE_RATE * 2 * channels;

    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    header[4] = (byte)(totalDataLenPlusHeader & 0xff);
    header[5] = (byte)((totalDataLenPlusHeader >> 8) & 0xff);
    header[6] = (byte)((totalDataLenPlusHeader >> 16) & 0xff);
    header[7] = (byte)((totalDataLenPlusHeader >> 24) & 0xff);
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
    header[20] = 1; header[21] = 0; // PCM format
    header[22] = (byte)channels; header[23] = 0;
    header[24] = (byte)(longSampleRate & 0xff);
    header[25] = (byte)((longSampleRate >> 8) & 0xff);
    header[26] = (byte)((longSampleRate >> 16) & 0xff);
    header[27] = (byte)((longSampleRate >> 24) & 0xff);
    header[28] = (byte)(byteRate & 0xff);
    header[29] = (byte)((byteRate >> 8) & 0xff);
    header[30] = (byte)((byteRate >> 16) & 0xff);
    header[31] = (byte)((byteRate >> 24) & 0xff);
    header[32] = (byte)(channels * 2); header[33] = 0;
    header[34] = 16; header[35] = 0; // 16-bit
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    header[40] = (byte)(totalAudioLen & 0xff);
    header[41] = (byte)((totalAudioLen >> 8) & 0xff);
    header[42] = (byte)((totalAudioLen >> 16) & 0xff);
    header[43] = (byte)((totalAudioLen >> 24) & 0xff);
}

// ---------------------------------------------------------------------------------
// 7. SPEAKER TEST TONE (Cautiously increased from 6000 to 11000)
// ---------------------------------------------------------------------------------
void testSpeaker() {
    Serial.printf("\n🔊 [MAX98357A] Running speaker hardware self-test (440Hz tone @ amplitude %d)...\n", TEST_TONE_AMPLITUDE);
    const int freq = 440;
    const int num_samples = SAMPLE_RATE / 2; // 0.5s duration
    int16_t stereo_buffer[512]; // 256 stereo pairs
    size_t bytes_written = 0;

    int sample_idx = 0;
    while (sample_idx < num_samples) {
        int chunk_samples = min(256, num_samples - sample_idx);
        for (int i = 0; i < chunk_samples; i++) {
            float t = (float)(sample_idx + i) / SAMPLE_RATE;
            int16_t val = (int16_t)(sin(2 * 3.14159265 * freq * t) * TEST_TONE_AMPLITUDE);
            stereo_buffer[2 * i]     = val; // Left
            stereo_buffer[2 * i + 1] = val; // Right
        }
        i2s_write(I2S_SPK_PORT, stereo_buffer, chunk_samples * 2 * sizeof(int16_t), &bytes_written, portMAX_DELAY);
        sample_idx += chunk_samples;
    }
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    Serial.println("✓ [MAX98357A] Speaker hardware self-test complete.");
}

// ---------------------------------------------------------------------------------
// 8. VAD SPEECH ACTIVITY CHECK
// ---------------------------------------------------------------------------------
bool checkSpeechActivity(int32_t* outAvgEnergy) {
    if (millis() < vadCooldownUntil || isProcessingTurn) {
        return false;
    }

    int32_t slot_chunk[VAD_CHUNK_SAMPLES];
    size_t bytes_read = 0;
    esp_err_t err = i2s_read(I2S_MIC_PORT, (char*)slot_chunk, sizeof(slot_chunk), &bytes_read, 20 / portTICK_PERIOD_MS);
    if (err != ESP_OK || bytes_read == 0) return false;

    size_t slots_read = bytes_read / sizeof(int32_t);
    int32_t total_amplitude = 0;
    for (size_t i = 0; i < slots_read; i++) {
        int16_t pcm16 = convertSlot32ToPcm16(slot_chunk[i]);
        total_amplitude += abs((int32_t)pcm16);
    }
    int32_t avg_amp = (slots_read > 0) ? (total_amplitude / slots_read) : 0;
    if (outAvgEnergy) *outAvgEnergy = avg_amp;

    return (avg_amp > SPEECH_ENERGY_THRESHOLD);
}

// ---------------------------------------------------------------------------------
// 9. CONTINUOUS UTTERANCE RECORDING WITH SILENCE VAD
// ---------------------------------------------------------------------------------
size_t recordUtterance(int16_t* dest_buffer, size_t max_samples, unsigned long max_record_ms, unsigned long silence_timeout_ms) {
    transitionTo(STATE_RECORDING, "speech_started");
    Serial.printf("🎙️ [MIC] Recording utterance (max %lu ms, silence cutoff %lu ms)...\n", max_record_ms, silence_timeout_ms);

    const size_t CHUNK_SLOTS = VAD_CHUNK_SAMPLES;
    int32_t slot_chunk[CHUNK_SLOTS];
    size_t samples_recorded = 0;
    size_t bytes_read = 0;

    bool speech_detected = false;
    unsigned long speech_start_time = 0;
    unsigned long silence_start_time = 0;
    unsigned long start_time = millis();

    int16_t min_sample = 32767;
    int16_t max_sample = -32768;
    int64_t sum_amp = 0;

    while (samples_recorded < max_samples && (millis() - start_time) < max_record_ms) {
        size_t slots_to_read = min(CHUNK_SLOTS, max_samples - samples_recorded);
        esp_err_t err = i2s_read(I2S_MIC_PORT, (char*)slot_chunk, slots_to_read * sizeof(int32_t), &bytes_read, 30 / portTICK_PERIOD_MS);
        if (err != ESP_OK || bytes_read == 0) {
            delay(2);
            continue;
        }

        size_t slots_read = bytes_read / sizeof(int32_t);
        int32_t chunk_sum = 0;

        for (size_t i = 0; i < slots_read; i++) {
            int16_t pcm16 = convertSlot32ToPcm16(slot_chunk[i]);
            dest_buffer[samples_recorded++] = pcm16;

            if (pcm16 < min_sample) min_sample = pcm16;
            if (pcm16 > max_sample) max_sample = pcm16;
            int32_t abs_val = abs((int32_t)pcm16);
            sum_amp += abs_val;
            chunk_sum += abs_val;
        }

        int32_t chunk_avg = (slots_read > 0) ? (chunk_sum / slots_read) : 0;

        if (chunk_avg > SPEECH_ENERGY_THRESHOLD) {
            speech_detected = true;
            silence_start_time = 0;
            if (speech_start_time == 0) speech_start_time = millis();
        } else if (speech_detected) {
            if (silence_start_time == 0) {
                silence_start_time = millis();
            } else if ((millis() - silence_start_time) >= silence_timeout_ms) {
                Serial.printf("  [VAD] End of speech detected (silence duration: %lu ms)\n", millis() - silence_start_time);
                break;
            }
        }
    }

    int32_t avg_amp = (samples_recorded > 0) ? (int32_t)(sum_amp / samples_recorded) : 0;
    Serial.printf("✓ [MIC] Capture finished: %u samples (%lu ms), Avg Amp: %d [Min: %d, Max: %d]\n",
                  samples_recorded, millis() - start_time, avg_amp, min_sample, max_sample);
    return samples_recorded;
}

// ---------------------------------------------------------------------------------
// 10. HTTP CLOUD COMMUNICATOR & STREAMING AUDIO PLAYBACK
// ---------------------------------------------------------------------------------
bool sendAudioAndProcessResponse(size_t pcm_samples, bool in_standby) {
    if (pcm_samples == 0) return false;
    transitionTo(STATE_PROCESSING, "uploading_audio");

    size_t pcm_bytes = pcm_samples * sizeof(int16_t);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("❌ [HTTP] Wi-Fi disconnected; cannot reach Mizo server.");
        return false;
    }

    WiFiClient client;
    client.setTimeout(25);

    Serial.printf("📡 [HTTP] Connecting to Mizo Cloud Server at %s:%d...\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
    if (!client.connect(MIZO_SERVER_IP, MIZO_SERVER_PORT)) {
        Serial.printf("❌ [HTTP] Connection failed to %s:%d\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
        return false;
    }

    String boundary = "----MizoESP32Boundary" + String(millis());
    byte wavHeader[44];
    setupWavHeader(wavHeader, pcm_bytes);

    String head = "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"audio\"; filename=\"inmp441.wav\"\r\n" +
                  "Content-Type: audio/wav\r\n\r\n";
    String tail = "\r\n--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"student_id\"\r\n\r\n1\r\n" +
                  "--" + boundary + "--\r\n";

    int totalPayloadLen = head.length() + 44 + pcm_bytes + tail.length();

    client.print("POST /api/v1/esp32/audio?format=audio HTTP/1.1\r\n");
    client.printf("Host: %s:%d\r\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
    client.printf("X-Device-Key: %s\r\n", DEVICE_API_KEY);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    client.printf("Content-Length: %d\r\n", totalPayloadLen);
    client.print("Connection: close\r\n\r\n");

    client.print(head);
    client.write(wavHeader, 44);

    size_t offset = 0;
    while (offset < pcm_bytes) {
        size_t chunk = min((size_t)1024, pcm_bytes - offset);
        client.write(record_buffer + offset, chunk);
        offset += chunk;
    }
    client.print(tail);
    client.flush();

    Serial.println("  [HTTP] Uploaded WAV payload. Waiting for server response...");

    // Wait for HTTP response headers (30s limit for Whisper STT & LLM)
    unsigned long timeout = millis() + 30000;
    while (client.connected() && !client.available()) {
        if (millis() > timeout) {
            Serial.println("❌ [HTTP] Timeout waiting for server response.");
            client.stop();
            return false;
        }
        delay(10);
    }

    // Read status line
    String statusLine = client.readStringUntil('\n');
    statusLine.trim();
    Serial.printf("📥 [HTTP STATUS] %s\n", statusLine.c_str());

    int statusCode = 0;
    int firstSpace = statusLine.indexOf(' ');
    if (firstSpace > 0) {
        statusCode = statusLine.substring(firstSpace + 1, firstSpace + 4).toInt();
    }

    // Parse HTTP headers
    String contentType = "";
    long contentLength = -1;
    String headerVoiceState = "";
    String headerTranscript = "";

    while (client.connected()) {
        String line = client.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) break;

        int colon = line.indexOf(':');
        if (colon > 0) {
            String name = line.substring(0, colon);
            String val  = line.substring(colon + 1);
            name.trim(); val.trim();
            name.toLowerCase();

            if (name == "content-type") {
                contentType = val;
                contentType.toLowerCase();
            } else if (name == "content-length") {
                contentLength = val.toInt();
            } else if (name == "x-voice-state") {
                headerVoiceState = val;
            } else if (name == "x-transcript") {
                headerTranscript = val;
            }
        }
    }

    Serial.printf("📄 [CONTENT TYPE] %s | [CONTENT LENGTH] %ld bytes\n", contentType.c_str(), contentLength);
    if (headerTranscript.length() > 0) {
        Serial.printf("📝 [TRANSCRIPTION] \"%s\"\n", headerTranscript.c_str());
    }

    if (statusCode != 200) {
        Serial.printf("❌ [HTTP ERROR] Non-200 code: %d\n", statusCode);
        String errBody = client.readString();
        Serial.printf("  Error Body: %s\n", errBody.c_str());
        client.stop();
        return false;
    }

    // A. JSON Response (Standby ignored, Stop command, or Metadata)
    if (contentType.indexOf("application/json") >= 0 || contentType.indexOf("audio") < 0) {
        String jsonBody = client.readString();
        client.stop();

        // Extract transcript
        String bodyTranscript = "";
        int tIdx = jsonBody.indexOf("\"transcript\":");
        if (tIdx >= 0) {
            int q1 = jsonBody.indexOf('"', tIdx + 13);
            if (q1 >= 0) {
                int q2 = jsonBody.indexOf('"', q1 + 1);
                if (q2 >= 0) bodyTranscript = jsonBody.substring(q1 + 1, q2);
            }
        }
        if (bodyTranscript.length() > 0) {
            Serial.printf("📝 [TRANSCRIPTION] \"%s\"\n", bodyTranscript.c_str());
        }

        // Check if Stop command was detected
        if (jsonBody.indexOf("\"command\":\"stop\"") >= 0 || jsonBody.indexOf("\"command\": \"stop\"") >= 0) {
            Serial.println("🛑 [STOP COMMAND DETECTED] Server confirmed 'Stop Mizo'. Halting conversation.");
            transitionTo(STATE_STOPPING, "stop_command_received");
            transitionTo(STATE_STANDBY, "stopped_to_standby");
            return true;
        }

        // Check if speech was ignored in Standby
        if (jsonBody.indexOf("\"ignored\":true") >= 0 || jsonBody.indexOf("\"ignored\": true") >= 0) {
            Serial.println("💤 [STANDBY] Speech ignored (no wake phrase 'Hey Mizo'). Remaining in STANDBY.");
            transitionTo(STATE_STANDBY, "no_wake_word");
            return true;
        }

        Serial.println("ℹ️ [RESPONSE] JSON metadata received.");
        return true;
    }

    // B. Audio Stream Response (WAV reply to play)
    Serial.println("🔊 [RESPONSE] Valid audio/wav stream received! Inspecting WAV header...");
    uint8_t wavHdr[44];
    size_t hdrRead = 0;
    unsigned long hdrTimeout = millis() + 5000;
    while (hdrRead < 44 && client.connected() && millis() < hdrTimeout) {
        if (client.available()) {
            wavHdr[hdrRead++] = client.read();
        } else {
            delay(2);
        }
    }

    if (hdrRead < 44) {
        Serial.println("❌ [WAV] Incomplete WAV header received.");
        client.stop();
        return false;
    }

    if (wavHdr[0] != 'R' || wavHdr[1] != 'I' || wavHdr[2] != 'F' || wavHdr[3] != 'F' ||
        wavHdr[8] != 'W' || wavHdr[9] != 'A' || wavHdr[10] != 'V' || wavHdr[11] != 'E') {
        Serial.println("❌ [WAV] Invalid magic bytes (Not a RIFF/WAVE stream).");
        client.stop();
        return false;
    }

    uint16_t audioFormat   = wavHdr[20] | (wavHdr[21] << 8);
    uint16_t numChannels   = wavHdr[22] | (wavHdr[23] << 8);
    uint32_t wavSampleRate = wavHdr[24] | (wavHdr[25] << 8) | (wavHdr[26] << 16) | (wavHdr[27] << 24);
    uint16_t bitsPerSample = wavHdr[34] | (wavHdr[35] << 8);

    Serial.printf("✓ [WAV] Format=%u (PCM), Channels=%u, Rate=%u Hz, Bits=%u\n",
                  audioFormat, numChannels, wavSampleRate, bitsPerSample);

    // Stream PCM to MAX98357A speaker with gain & saturation protection
    transitionTo(STATE_SPEAKING, "playing_reply_audio");
    Serial.printf("🔊 [PLAYBACK] Streaming audio through MAX98357A speaker (Gain: %.2fx, Dual-Drive)...\n", PLAYBACK_GAIN);

    uint8_t rawChunk[512]; // Up to 256 mono samples
    int16_t stereoBuffer[512]; // 256 stereo pairs
    size_t total_played = 0;
    unsigned long streamWatchdog = millis() + 25000;

    while ((client.connected() || client.available()) && millis() < streamWatchdog) {
        size_t avail = client.available();
        if (avail > 0) {
            // Read even number of bytes for 16-bit PCM
            int toRead = min((int)avail, (int)sizeof(rawChunk));
            if (toRead % 2 != 0) toRead--;
            if (toRead <= 0) toRead = 2;

            int readBytes = client.readBytes((char*)rawChunk, toRead);
            if (readBytes > 0) {
                // Apply playback gain, saturation clamping, and drive both L and R channels
                int16_t* pcm_samples = (int16_t*)rawChunk;
                int count = readBytes / sizeof(int16_t);
                for (int i = 0; i < count; i++) {
                    int16_t sample = applyGainAndClamp(pcm_samples[i], PLAYBACK_GAIN);
                    stereoBuffer[2 * i]     = sample; // Left
                    stereoBuffer[2 * i + 1] = sample; // Right
                }

                size_t bytes_written = 0;
                i2s_write(I2S_SPK_PORT, stereoBuffer, count * 2 * sizeof(int16_t), &bytes_written, portMAX_DELAY);
                total_played += bytes_written;
                streamWatchdog = millis() + 5000;
            }
        } else {
            delay(2);
        }
    }

    client.stop();
    i2s_zero_dma_buffer(I2S_SPK_PORT);
    Serial.printf("✓ [PLAYBACK] Completed! Played %u bytes through speaker.\n", total_played);

    // Echo protection: flush microphone DMA buffer and set cooldown
    i2s_zero_dma_buffer(I2S_MIC_PORT);
    vadCooldownUntil = millis() + ECHO_COOLDOWN_MS;

    // Transition to ACTIVE_LISTENING for continuous conversation
    lastActiveTurnTime = millis();
    transitionTo(STATE_ACTIVE_LISTENING, "playback_completed_continue_conversation");
    return true;
}

// ---------------------------------------------------------------------------------
// 11. HEARTBEAT & WI-FI RECONNECT
// ---------------------------------------------------------------------------------
unsigned long lastWiFiRetry = 0;
void checkAndReconnectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    if (millis() - lastWiFiRetry < 10000UL) return;
    lastWiFiRetry = millis();
    Serial.printf("[Wi-Fi] Disconnected (status=%d). Retrying connection to '%s'...\n",
                  (int)WiFi.status(), WIFI_SSID);
    WiFi.disconnect(false, false);
    delay(100);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

unsigned long lastHeartbeat = 0;
void checkAndSendHeartbeat() {
    if (millis() - lastHeartbeat >= 20000) {
        lastHeartbeat = millis();
        if (WiFi.status() == WL_CONNECTED && !isProcessingTurn) {
            HTTPClient http;
            String url = String("http://") + MIZO_SERVER_IP + ":" + String(MIZO_SERVER_PORT) + "/api/v1/esp32/ping";
            http.begin(url);
            http.addHeader("X-Device-Key", DEVICE_API_KEY);
            int code = http.GET();
            http.end();
        }
    }
}

// ---------------------------------------------------------------------------------
// 12. SETUP
// ---------------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1200);

    Serial.println("\n=============================================================");
    Serial.println("    MIZO 3.0 — WAKE-WORD & CONTINUOUS CONVERSATION FIRMWARE  ");
    Serial.println("=============================================================");
    Serial.println("Hardware Architecture:");
    Serial.println("  INMP441 Microphone : SCK=GPIO4, WS=GPIO5, SD=GPIO6, L/R=GND");
    Serial.println("  MAX98357A Speaker  : BCLK=GPIO18, LRC=GPIO21, DIN=GPIO20");
    Serial.println("  Manual Trigger     : BOOT Button (GPIO 0)");
    Serial.printf("  Server Endpoint    : http://%s:%d/api/v1/esp32/audio?format=audio\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
    Serial.printf("  Playback Gain      : %.2fx with saturation clipping clamp\n", PLAYBACK_GAIN);
    Serial.println("=============================================================");

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // Allocate 16-bit PCM recording buffer (192,000 bytes for 6 seconds @ 16kHz)
    record_buffer = (uint8_t*)ps_malloc(BUFFER_SIZE_BYTES);
    if (!record_buffer) {
        record_buffer = (uint8_t*)malloc(BUFFER_SIZE_BYTES);
    }
    if (!record_buffer) {
        Serial.println("❌ Failed to allocate recording buffer!");
        while (1) delay(1000);
    }
    Serial.printf("✓ Audio recording buffer allocated: %u bytes (%d sec @ 16kHz)\n", BUFFER_SIZE_BYTES, MAX_RECORD_SECONDS);

    setupI2SMicrophone();
    setupI2SSpeaker();

    // 1. Run 440Hz test tone at startup (amplitude 11000)
    testSpeaker();

    // 2. Connect Wi-Fi
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.persistent(false);
    WiFi.setAutoReconnect(true);

    Serial.printf("\n[Wi-Fi] Connecting to '%s'...\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000UL) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("✓ [Wi-Fi] Connected successfully!");
        Serial.print("  ESP32 IP: "); Serial.println(WiFi.localIP());
        Serial.printf("  Signal: %d dBm\n", WiFi.RSSI());
    } else {
        Serial.println("⚠️ [Wi-Fi] Connection timed out. Auto-reconnect will retry in loop().");
    }

    // 3. Enter STANDBY mode at startup
    transitionTo(STATE_STANDBY, "boot_initialization");
    Serial.println("\n=============================================================");
    Serial.println("🤖 MIZO IS READY IN STANDBY MODE");
    Serial.println("Say 'Hey Mizo' or 'Hi Mizo' into the microphone to wake up!");
    Serial.println("Say 'Mizo stop' or 'Stop Mizo' during conversation to return to STANDBY.");
    Serial.println("Or press BOOT Button (GPIO 0) / send 'w' or 't' via Serial Monitor.");
    Serial.println("=============================================================\n");
}

// ---------------------------------------------------------------------------------
// 13. MAIN LOOP & STATE MACHINE
// ---------------------------------------------------------------------------------
int consecutiveSpeechHits = 0;

void loop() {
    static bool previousButtonState = HIGH;
    static bool buttonArmed = true;

    // Background housekeeping
    checkAndReconnectWiFi();
    checkAndSendHeartbeat();

    // A. Manual Button Trigger (GPIO 0 BOOT Button)
    bool buttonState = digitalRead(BUTTON_PIN);
    if (buttonState == HIGH) buttonArmed = true;
    if (buttonArmed && previousButtonState == HIGH && buttonState == LOW) {
        delay(35);
        if (digitalRead(BUTTON_PIN) == LOW) {
            buttonArmed = false;
            Serial.println("\n🔘 [BUTTON] BOOT button pressed!");
            if (currentState == STATE_STANDBY) {
                transitionTo(STATE_ACTIVE_LISTENING, "boot_button_wake");
            }
            if (!isProcessingTurn) {
                isProcessingTurn = true;
                size_t samples = recordUtterance((int16_t*)record_buffer, TOTAL_MAX_SAMPLES, ACTIVE_MAX_RECORD_MS, SILENCE_TIMEOUT_MS);
                sendAudioAndProcessResponse(samples, false);
                isProcessingTurn = false;
            }
            previousButtonState = digitalRead(BUTTON_PIN);
            return;
        }
    }
    previousButtonState = buttonState;

    // B. Serial Monitor Commands ('w'=wake, 's'=stop, 't'=turn test, 'p'=test tone)
    while (Serial.available() > 0) {
        char ch = (char)Serial.read();
        if (ch == 'w' || ch == 'W') {
            Serial.println("\n[SERIAL] 'w' command: Waking Mizo to ACTIVE_LISTENING.");
            transitionTo(STATE_ACTIVE_LISTENING, "serial_command_wake");
            return;
        } else if (ch == 's' || ch == 'S') {
            Serial.println("\n[SERIAL] 's' command: Stopping conversation -> STANDBY.");
            transitionTo(STATE_STANDBY, "serial_command_stop");
            return;
        } else if (ch == 'p' || ch == 'P') {
            Serial.println("\n[SERIAL] 'p' command: Running speaker test tone.");
            testSpeaker();
            return;
        } else if (ch == 't' || ch == 'T') {
            Serial.println("\n[SERIAL] 't' command: Triggering recording turn.");
            if (!isProcessingTurn) {
                isProcessingTurn = true;
                size_t samples = recordUtterance((int16_t*)record_buffer, TOTAL_MAX_SAMPLES, ACTIVE_MAX_RECORD_MS, SILENCE_TIMEOUT_MS);
                sendAudioAndProcessResponse(samples, (currentState == STATE_STANDBY));
                isProcessingTurn = false;
            }
            return;
        }
    }

    // C. Conversation Inactivity Timeout in ACTIVE_LISTENING
    if (currentState == STATE_ACTIVE_LISTENING && !isProcessingTurn) {
        if (millis() - lastActiveTurnTime >= INACTIVITY_TIMEOUT_MS) {
            Serial.printf("\n⌛ [TIMEOUT] Inactivity timeout (%u s). No speech heard. Returning to STANDBY.\n",
                          INACTIVITY_TIMEOUT_MS / 1000);
            transitionTo(STATE_STANDBY, "inactivity_timeout");
        }
    }

    // D. Speech Activity Detection & Turn Trigger
    if (!isProcessingTurn && millis() >= vadCooldownUntil) {
        int32_t currentEnergy = 0;
        bool speechHit = checkSpeechActivity(&currentEnergy);

        if (speechHit) {
            consecutiveSpeechHits++;
            if (consecutiveSpeechHits >= REQUIRED_SPEECH_HITS) {
                consecutiveSpeechHits = 0;
                isProcessingTurn = true;

                if (currentState == STATE_STANDBY) {
                    Serial.printf("\n🎙️ [STANDBY VAD] Speech detected (Energy: %d). Listening for wake phrase 'Hey Mizo'...\n", currentEnergy);
                    // Record shorter window for wake word attempt
                    size_t samples = recordUtterance((int16_t*)record_buffer, TOTAL_MAX_SAMPLES, STANDBY_MAX_RECORD_MS, 900);
                    sendAudioAndProcessResponse(samples, true);
                } else if (currentState == STATE_ACTIVE_LISTENING) {
                    Serial.printf("\n🎙️ [ACTIVE VAD] User speech detected (Energy: %d). Listening to conversational turn...\n", currentEnergy);
                    // Record full conversational utterance with silence VAD
                    size_t samples = recordUtterance((int16_t*)record_buffer, TOTAL_MAX_SAMPLES, ACTIVE_MAX_RECORD_MS, SILENCE_TIMEOUT_MS);
                    sendAudioAndProcessResponse(samples, false);
                }

                isProcessingTurn = false;
                return;
            }
        } else {
            consecutiveSpeechHits = 0;
        }
    }

    delay(20);
}
