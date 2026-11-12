/*

 * =================================================================================

 * MIZO 3.0 — ESP32-S3 ROBOTIC TUTOR FIRMWARE

 * Mode: MICROPHONE TRANSCRIPTION & SPEAKER TEST MODE

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

 *   VIN  --> 5V (or 3.3V)

 *   GND  --> GND

 *   BCLK --> GPIO 18 (Bit Clock)

 *   LRC  --> GPIO 21 (Left/Right Clock / WS)

 *   DIN  --> GPIO 20 (Data In)

 *   GAIN --> GND (9dB gain)

 *

 * TEST TRIGGER (Hardware Button):

 *   BOOT Button --> GPIO 0 (Press to trigger 5-second record & transcribe test)

 * =================================================================================

 */



#include <WiFi.h>

#include <HTTPClient.h>

#include <driver/i2s.h>

#include <math.h>



// ---------------------------------------------------------------------------------

// LOCAL CREDENTIALS & SECRETS CONFIGURATION

// ---------------------------------------------------------------------------------

// Reuses credentials from secrets.h in this same directory.

// secrets.h is excluded from Git via .gitignore to keep credentials private.

#if __has_include("secrets.h")

  #include "secrets.h"

  const char* WIFI_SSID     = SECRET_WIFI_SSID;

  const char* WIFI_PASSWORD = SECRET_WIFI_PASSWORD;

#else

  const char* WIFI_SSID     = "MizoTest";

  const char* WIFI_PASSWORD = "";

#endif



// Mizo Cloud Server Configuration

const char* MIZO_SERVER_IP   = "10.211.4.52";

const int   MIZO_SERVER_PORT = 8000;



// Device API key (Configured in Mizo Admin)

const char* DEVICE_API_KEY   = "mizo_esp32_secret_key_123";



// ---------------------------------------------------------------------------------

// AUDIO CONFIGURATION

// ---------------------------------------------------------------------------------

#define SAMPLE_RATE         16000

#define RECORD_SECONDS      5      // 5-second test capture

#define TOTAL_SAMPLES       (SAMPLE_RATE * RECORD_SECONDS)

#define BUFFER_SIZE_BYTES   (TOTAL_SAMPLES * sizeof(int16_t)) // 16-bit Mono (160,000 bytes)



// I2S Microphone Pins (INMP441)

#define I2S_MIC_PORT        I2S_NUM_0

#define I2S_MIC_SCK         4   // SCK (Serial Clock / BCLK)

#define I2S_MIC_WS          5   // WS  (Word Select / LRCLK)

#define I2S_MIC_SD          6   // SD  (Serial Data from Mic)



// I2S Speaker Pins (MAX98357A)

#define I2S_SPK_PORT        I2S_NUM_1

#define I2S_SPK_BCLK        18  // BCLK (Bit Clock)

#define I2S_SPK_LRC         21  // LRC  (Word Select / LRCLK)

#define I2S_SPK_DIN         20  // DIN  (Data In to DAC)



// BOOT button on ESP32-S3 DevKit

#define BUTTON_PIN          0



// Voice Activity Detection (VAD) parameters for standby mode

#define VAD_CHUNK_SAMPLES          256

#define SPEECH_THRESHOLD           1100

#define REQUIRED_CONSECUTIVE_HITS  2

#define VAD_COOLDOWN_MS            1500



// Recording buffer in PSRAM or Heap

uint8_t* record_buffer = NULL;

unsigned long vad_cooldown_until = 0;

int consecutive_speech_hits = 0;

bool is_test_running = false;



// ---------------------------------------------------------------------------------

// 32-BIT TO 16-BIT SAMPLE CONVERSION

// ---------------------------------------------------------------------------------

// The INMP441 MEMS microphone transmits 24-bit audio inside a 32-bit I2S slot.

// Shifting right by 14 provides clean standard gain (+12 dB over 16-bit truncation)

// while strict saturation clamping [-32768, 32767] prevents integer overflow.

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

// I2S DRIVER INITIALIZATION

// ---------------------------------------------------------------------------------

void setupI2SMicrophone() {

    i2s_config_t i2s_config = {

        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),

        .sample_rate = SAMPLE_RATE,

        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // Critical: 32-bit slot for INMP441

        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),

        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 4,

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

}



// ---------------------------------------------------------------------------------

// WAV HEADER GENERATION (16kHz, 16-bit Mono RIFF WAV)

// ---------------------------------------------------------------------------------

void setupWavHeader(byte* header, int totalDataLen) {

    long totalAudioLen = totalDataLen;

    long totalDataLenPlusHeader = totalAudioLen + 36;

    long longSampleRate = SAMPLE_RATE;

    int channels = 1;

    long byteRate = SAMPLE_RATE * 2 * channels; // 16-bit = 2 bytes per sample



    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';

    header[4] = (byte)(totalDataLenPlusHeader & 0xff);

    header[5] = (byte)((totalDataLenPlusHeader >> 8) & 0xff);

    header[6] = (byte)((totalDataLenPlusHeader >> 16) & 0xff);

    header[7] = (byte)((totalDataLenPlusHeader >> 24) & 0xff);

    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';

    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';

    header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0; // 16 bytes for PCM

    header[20] = 1; header[21] = 0;                                  // AudioFormat 1 = PCM

    header[22] = (byte)channels; header[23] = 0;

    header[24] = (byte)(longSampleRate & 0xff);

    header[25] = (byte)((longSampleRate >> 8) & 0xff);

    header[26] = (byte)((longSampleRate >> 16) & 0xff);

    header[27] = (byte)((longSampleRate >> 24) & 0xff);

    header[28] = (byte)(byteRate & 0xff);

    header[29] = (byte)((byteRate >> 8) & 0xff);

    header[30] = (byte)((byteRate >> 16) & 0xff);

    header[31] = (byte)((byteRate >> 24) & 0xff);

    header[32] = (byte)(channels * 2); header[33] = 0;               // BlockAlign (2 bytes)

    header[34] = 16; header[35] = 0;                                 // BitsPerSample (16 bits)

    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';

    header[40] = (byte)(totalAudioLen & 0xff);

    header[41] = (byte)((totalAudioLen >> 8) & 0xff);

    header[42] = (byte)((totalAudioLen >> 16) & 0xff);

    header[43] = (byte)((totalAudioLen >> 24) & 0xff);

}



// ---------------------------------------------------------------------------------

// SPEAKER HARDWARE AUDIO SELF-TEST

// ---------------------------------------------------------------------------------

void testSpeaker() {
    Serial.println("\n🔊 [MAX98357A] Running speaker hardware audio self-test (440Hz tone @ amplitude 28000)...");
    const int freq = 440;
    const int num_samples = SAMPLE_RATE / 2; // 0.5s duration
    int16_t stereo_buffer[512]; // 256 stereo pairs
    size_t bytes_written = 0;

    int sample_idx = 0;
    while (sample_idx < num_samples) {
        int chunk_samples = min(256, num_samples - sample_idx);
        for (int i = 0; i < chunk_samples; i++) {
            float t = (float)(sample_idx + i) / SAMPLE_RATE;
            int16_t val = (int16_t)(sin(2 * 3.14159265 * freq * t) * 28000);
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

// INMP441 RECORDING (5 SECONDS, 16kHz, 16-BIT MONO)

// ---------------------------------------------------------------------------------

void recordSpeechAudio(int16_t* dest_buffer, size_t total_samples) {

    const size_t CHUNK_SLOTS = 256;

    int32_t slot_chunk[CHUNK_SLOTS];

    size_t samples_recorded = 0;

    size_t bytes_read = 0;



    int16_t min_sample = 32767;

    int16_t max_sample = -32768;

    int64_t sum_amp = 0;



    unsigned long start_time = millis();

    int last_sec = -1;



    while (samples_recorded < total_samples) {

        int elapsed_sec = (millis() - start_time) / 1000;

        if (elapsed_sec != last_sec && elapsed_sec < RECORD_SECONDS) {

            last_sec = elapsed_sec;

            Serial.printf("  ... recording [%d/%d seconds]\n", elapsed_sec + 1, RECORD_SECONDS);

        }



        size_t slots_to_read = min(CHUNK_SLOTS, total_samples - samples_recorded);

        esp_err_t err = i2s_read(I2S_MIC_PORT, (char*)slot_chunk, slots_to_read * sizeof(int32_t), &bytes_read, portMAX_DELAY);

        if (err != ESP_OK || bytes_read == 0) {

            delay(2);

            continue;

        }



        size_t slots_read = bytes_read / sizeof(int32_t);

        for (size_t i = 0; i < slots_read; i++) {

            int16_t pcm16 = convertSlot32ToPcm16(slot_chunk[i]);

            dest_buffer[samples_recorded++] = pcm16;



            if (pcm16 < min_sample) min_sample = pcm16;

            if (pcm16 > max_sample) max_sample = pcm16;

            sum_amp += abs((int32_t)pcm16);

        }

    }



    int32_t avg_amp = (int32_t)(sum_amp / total_samples);

    Serial.printf("✓ Capture complete: %u samples (%u bytes PCM)\n", total_samples, total_samples * sizeof(int16_t));

    Serial.printf("  [Mic Signal Level] Min: %d, Max: %d, Avg Amp: %d\n", min_sample, max_sample, avg_amp);



    if (avg_amp < 30) {

        Serial.println("  ⚠️ Warning: Microphone signal level is very low. Please speak closely into the INMP441.");

    }

}



// ---------------------------------------------------------------------------------

// MICROPHONE TRANSCRIPTION & SPEAKER TEST RUNNER

// ---------------------------------------------------------------------------------

void runMicAndSpeakerTest() {

    if (is_test_running) return;

    is_test_running = true;



    Serial.println("\n=============================================================");

    Serial.println("🎙️ [TEST MODE] STARTING 5-SECOND RECORDING & TRANSCRIPTION TEST");

    Serial.println("=============================================================");

    Serial.println("Speak clearly into the INMP441 microphone now (e.g. 'Hello Mizo!')...");



    // 1. Record 5 seconds of audio from INMP441

    recordSpeechAudio((int16_t*)record_buffer, TOTAL_SAMPLES);

    size_t total_pcm_bytes = BUFFER_SIZE_BYTES;



    if (WiFi.status() != WL_CONNECTED) {

        Serial.println("❌ Wi-Fi disconnected! Cannot reach server.");

        is_test_running = false;

        return;

    }



    // 2. Connect to Mizo Server

    Serial.printf("\n📡 Connecting to Mizo Server at %s:%d...\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);

    WiFiClient client;

    client.setTimeout(15);

    if (!client.connect(MIZO_SERVER_IP, MIZO_SERVER_PORT)) {

        Serial.printf("❌ Could not connect to Mizo Cloud Server at %s:%d\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);

        is_test_running = false;

        return;

    }



    // 3. Prepare multipart form-data payload with 44-byte WAV header

    String boundary = "----MizoESP32Boundary" + String(millis());

    byte wavHeader[44];

    setupWavHeader(wavHeader, total_pcm_bytes);



    String head = "--" + boundary + "\r\n" +

                  "Content-Disposition: form-data; name=\"audio\"; filename=\"inmp441.wav\"\r\n" +

                  "Content-Type: audio/wav\r\n\r\n";

    String tail = "\r\n--" + boundary + "\r\n" +

                  "Content-Disposition: form-data; name=\"student_id\"\r\n\r\n1\r\n" +

                  "--" + boundary + "--\r\n";



    int totalPayloadLen = head.length() + 44 + total_pcm_bytes + tail.length();



    Serial.println("📡 Uploading WAV stream to /api/v1/esp32/audio?format=audio...");

    client.print("POST /api/v1/esp32/audio?format=audio HTTP/1.1\r\n");

    client.printf("Host: %s:%d\r\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);

    client.printf("X-Device-Key: %s\r\n", DEVICE_API_KEY);

    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());

    client.printf("Content-Length: %d\r\n", totalPayloadLen);

    client.print("Connection: close\r\n\r\n");



    // Send multipart prefix

    client.print(head);



    // Send valid 44-byte WAV header

    client.write(wavHeader, 44);



    // Stream 16-bit PCM in 1024-byte chunks

    size_t offset = 0;

    while (offset < total_pcm_bytes) {

        size_t chunk = min((size_t)1024, total_pcm_bytes - offset);

        client.write(record_buffer + offset, chunk);

        offset += chunk;

    }



    // Send multipart suffix

    client.print(tail);

    client.flush();

    Serial.println("✓ Audio uploaded! Waiting for Mizo server response (transcription & audio)...");



    // 4. Wait for HTTP response headers (30s timeout for Whisper STT & LLM)

    unsigned long timeout = millis() + 30000;

    while (client.connected() && !client.available()) {

        if (millis() > timeout) {

            Serial.println("❌ Timeout waiting for server response (30s limit reached).");

            client.stop();

            is_test_running = false;

            return;

        }

        delay(10);

    }



    // 5. Read and print HTTP status line

    String statusLine = client.readStringUntil('\n');

    statusLine.trim();

    Serial.printf("\n-------------------------------------------------------------\n");

    Serial.printf("📥 [HTTP STATUS] %s\n", statusLine.c_str());



    int statusCode = 0;

    int firstSpace = statusLine.indexOf(' ');

    if (firstSpace > 0) {

        statusCode = statusLine.substring(firstSpace + 1, firstSpace + 4).toInt();

    }



    // 6. Parse and print HTTP response headers

    String contentType = "";

    long contentLength = -1;

    while (client.connected()) {

        String line = client.readStringUntil('\n');

        line.trim();

        if (line.length() == 0) break; // End of HTTP headers



        int colon = line.indexOf(':');

        if (colon > 0) {

            String headerName = line.substring(0, colon);

            String headerVal  = line.substring(colon + 1);

            headerName.trim();

            headerVal.trim();

            headerName.toLowerCase();

            if (headerName == "content-type") {

                contentType = headerVal;

                contentType.toLowerCase();

            } else if (headerName == "content-length") {

                contentLength = headerVal.toInt();

            }

        }

    }



    Serial.printf("📄 [CONTENT TYPE] %s\n", contentType.c_str());

    Serial.printf("📦 [CONTENT LENGTH] %ld bytes\n", contentLength);



    // 7. Check for HTTP errors

    if (statusCode != 200) {

        Serial.printf("❌ Server returned non-200 HTTP code: %d\n", statusCode);

        String errBody = client.readString();

        Serial.println("Error Body:");

        Serial.println(errBody);

        client.stop();

        is_test_running = false;

        return;

    }



    // 8. Handle JSON response (e.g. ignored in standby, control command, or metadata)

    if (contentType.indexOf("application/json") >= 0 || contentType.indexOf("audio") < 0) {

        Serial.println("ℹ️ [RESPONSE FORMAT] Server returned JSON data (No audio response to play).");

        String jsonBody = client.readString();

        Serial.println("\n--- RAW JSON BODY ---");

        Serial.println(jsonBody);

        Serial.println("---------------------");



        // Parse and print transcription text if present

        int tIdx = jsonBody.indexOf("\"transcript\":");

        if (tIdx >= 0) {

            int startQuote = jsonBody.indexOf('"', tIdx + 13);

            if (startQuote >= 0) {

                int endQuote = jsonBody.indexOf('"', startQuote + 1);

                if (endQuote >= 0) {

                    String transcript = jsonBody.substring(startQuote + 1, endQuote);

                    Serial.printf("\n📝 [TRANSCRIPTION] \"%s\"\n", transcript.c_str());

                }

            }

        }



        // Parse and print reply text if present

        int rIdx = jsonBody.indexOf("\"reply_text\":");

        if (rIdx >= 0) {

            int startQuote = jsonBody.indexOf('"', rIdx + 13);

            if (startQuote >= 0) {

                int endQuote = jsonBody.indexOf('"', startQuote + 1);

                if (endQuote >= 0) {

                    String reply = jsonBody.substring(startQuote + 1, endQuote);

                    Serial.printf("🤖 [MIZO REPLY]    \"%s\"\n", reply.c_str());

                }

            }

        }



        client.stop();

        Serial.println("\n✓ Test finished. Press BOOT button (GPIO 0) to run again.");

        is_test_running = false;

        return;

    }



    // 9. Handle WAV Audio stream response

    Serial.println("\n🔊 [RESPONSE FORMAT] Valid audio/wav stream detected! Inspecting WAV header...");

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

        Serial.println("❌ Incomplete WAV header received from server!");

        client.stop();

        is_test_running = false;

        return;

    }



    // Inspect RIFF and WAVE magic bytes

    if (wavHdr[0] != 'R' || wavHdr[1] != 'I' || wavHdr[2] != 'F' || wavHdr[3] != 'F' ||

        wavHdr[8] != 'W' || wavHdr[9] != 'A' || wavHdr[10] != 'V' || wavHdr[11] != 'E') {

        Serial.println("❌ Invalid WAV header (Missing RIFF/WAVE magic bytes)! Aborting playback.");

        client.stop();

        is_test_running = false;

        return;

    }



    uint16_t audioFormat   = wavHdr[20] | (wavHdr[21] << 8);

    uint16_t numChannels   = wavHdr[22] | (wavHdr[23] << 8);

    uint32_t wavSampleRate = wavHdr[24] | (wavHdr[25] << 8) | (wavHdr[26] << 16) | (wavHdr[27] << 24);

    uint16_t bitsPerSample = wavHdr[34] | (wavHdr[35] << 8);



    Serial.printf("✓ Valid WAV Header: Format=%u (PCM), Channels=%u, Rate=%u Hz, Bits=%u\n",

                  audioFormat, numChannels, wavSampleRate, bitsPerSample);



    // Stream PCM payload directly to MAX98357A I2S DAC speaker

    Serial.println("🔊 Streaming audio reply through MAX98357A speaker...");

    uint8_t audioChunk[1024];

    size_t bytes_written = 0;

    size_t total_played = 0;

    unsigned long streamWatchdog = millis() + 25000;



    while ((client.connected() || client.available()) && millis() < streamWatchdog) {

        size_t avail = client.available();

        if (avail > 0) {
            int toRead = min((int)avail, (int)sizeof(audioChunk));
            if (toRead % 2 != 0) toRead--;
            if (toRead <= 0) toRead = 2;

            int readBytes = client.readBytes((char*)audioChunk, toRead);
            if (readBytes > 0) {
                int16_t* pcm_samples = (int16_t*)audioChunk;
                int count = readBytes / sizeof(int16_t);
                int16_t stereoBuffer[512];
                for (int i = 0; i < count; i++) {
                    int16_t sample = applyGainAndClamp(pcm_samples[i], 1.50f);
                    stereoBuffer[2 * i]     = sample; // Left
                    stereoBuffer[2 * i + 1] = sample; // Right
                }
                i2s_write(I2S_SPK_PORT, stereoBuffer, count * 2 * sizeof(int16_t), &bytes_written, portMAX_DELAY);
                total_played += bytes_written;
                streamWatchdog = millis() + 5000;
            }
        } else {

            delay(2);

        }

    }

    client.stop();

    Serial.printf("✓ Audio playback complete! (Played %u bytes through speaker)\n", total_played);

    Serial.println("=============================================================");

    Serial.println(">>> Test finished. Press BOOT button (GPIO 0) to run again. <<<");

    Serial.println("=============================================================");



    vad_cooldown_until = millis() + VAD_COOLDOWN_MS;

    is_test_running = false;

}



// ---------------------------------------------------------------------------------

// CONTINUOUS MICROPHONE MONITORING (VAD on converted 16-bit PCM)

// ---------------------------------------------------------------------------------

bool detectSpeechActivity() {

    if (millis() < vad_cooldown_until || is_test_running) {

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

    int32_t avg_amplitude = total_amplitude / slots_read;



    if (avg_amplitude > SPEECH_THRESHOLD) {

        consecutive_speech_hits++;

        if (consecutive_speech_hits >= REQUIRED_CONSECUTIVE_HITS) {

            consecutive_speech_hits = 0;

            return true;

        }

    } else {

        consecutive_speech_hits = 0;

    }

    return false;

}



// ---------------------------------------------------------------------------------

// HEARTBEAT PING TO SERVER

// ---------------------------------------------------------------------------------

unsigned long lastWiFiRetry = 0;

void checkAndReconnectWiFi() {
    if (WiFi.status() == WL_CONNECTED) return;
    if (millis() - lastWiFiRetry < 10000UL) return;

    lastWiFiRetry = millis();
    Serial.printf("[Wi-Fi] Disconnected (status=%d). Retrying hotspot connection...\\n",
                  (int)WiFi.status());
    WiFi.disconnect(false, false);
    delay(100);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

unsigned long lastHeartbeat = 0;



void checkAndSendHeartbeat() {

    if (millis() - lastHeartbeat >= 20000) {

        lastHeartbeat = millis();

        if (WiFi.status() == WL_CONNECTED && !is_test_running) {

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

// ARDUINO SETUP & LOOP

// ---------------------------------------------------------------------------------

void setup() {

    Serial.begin(115200);

    delay(1000);



    Serial.println("\n=============================================================");

    Serial.println("   MIZO 3.0 — MICROPHONE TRANSCRIPTION & SPEAKER TEST MODE   ");

    Serial.println("=============================================================");

    Serial.println("Hardware Setup:");

    Serial.println("  INMP441 Microphone : SCK=GPIO4, WS=GPIO5, SD=GPIO6, L/R=GND");

    Serial.println("  MAX98357A Speaker  : BCLK=GPIO18, LRC=GPIO21, DIN=GPIO20");

    Serial.println("  Test Trigger       : BOOT Button (GPIO 0)");

    Serial.printf("  Server Target      : http://%s:%d/api/v1/esp32/audio?format=audio\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);

    Serial.println("=============================================================");



    pinMode(BUTTON_PIN, INPUT_PULLUP);



    // Allocate 16-bit PCM recording buffer (160,000 bytes for 5 seconds @ 16kHz)

    record_buffer = (uint8_t*)ps_malloc(BUFFER_SIZE_BYTES);

    if (!record_buffer) {

        record_buffer = (uint8_t*)malloc(BUFFER_SIZE_BYTES);

    }

    if (!record_buffer) {

        Serial.println("❌ Failed to allocate recording buffer!");

        while (1) delay(1000);

    }

    Serial.printf("✓ Audio buffer allocated (%u bytes)\n", BUFFER_SIZE_BYTES);



    // Initialize I2S peripherals

    setupI2SMicrophone();

    setupI2SSpeaker();



    // Run working speaker test on startup

    testSpeaker();



    // Connect to Wi-Fi with a timeout and diagnostic output.
    // If the first attempt fails, loop() retries periodically.
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.persistent(false);
    WiFi.setAutoReconnect(true);

    Serial.printf("\n[Wi-Fi] Connecting to SSID '%s'...\n", WIFI_SSID);
    Serial.println("[Wi-Fi] Check that the phone hotspot uses 2.4 GHz.");
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const unsigned long wifiStart = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 20000UL) {
        delay(500);
        Serial.printf("[Wi-Fi] status=%d, waiting...\n", (int)WiFi.status());
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("[Wi-Fi] Connected successfully.");
        Serial.print("[Wi-Fi] ESP32 IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("[Wi-Fi] Gateway: ");
        Serial.println(WiFi.gatewayIP());
        Serial.print("[Wi-Fi] Signal strength (RSSI): ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
    } else {
        Serial.println("[Wi-Fi] Connection timed out after 20 seconds.");
        Serial.println("[Wi-Fi] Verify SSID/password in secrets.h (if present).");
        Serial.println("[Wi-Fi] Try 2.4 GHz hotspot and WPA2-Personal security.");
        Serial.println("[Wi-Fi] Firmware will retry automatically in loop().");
    }



    Serial.println("\n=============================================================");

    Serial.println(">>> PRESS BOOT BUTTON (GPIO 0) TO RUN 5-SECOND TEST <<<");

    Serial.println("(Or send 't' in Serial Monitor to trigger test remotely)");

    Serial.println("Say: 'Hello Mizo' or any speech into the INMP441 microphone.");

    Serial.println("=============================================================\n");

}



void loop() {
    static bool previousButtonState = HIGH;
    static bool buttonArmed = true;

    // 0. Periodic heartbeat to server
    checkAndReconnectWiFi();

    checkAndSendHeartbeat();

    // 1. BOOT button: edge-triggered and non-blocking.
    // Do not wait forever for GPIO0 to be released.
    bool buttonState = digitalRead(BUTTON_PIN);
    if (buttonState == HIGH) {
        buttonArmed = true;
    }
    if (buttonArmed && previousButtonState == HIGH && buttonState == LOW) {
        delay(35); // debounce
        if (digitalRead(BUTTON_PIN) == LOW) {
            buttonArmed = false;
            Serial.println("\n[BUTTON] BOOT press confirmed; starting test.");
            runMicAndSpeakerTest();
            previousButtonState = digitalRead(BUTTON_PIN);
            return;
        }
    }
    previousButtonState = buttonState;

    // 2. Serial Monitor trigger: accept lowercase t or uppercase T.
    while (Serial.available() > 0) {
        int incoming = Serial.read();
        if (incoming < 0) break;
        char ch = (char)incoming;
        Serial.printf("[DEBUG] Serial character received: %d\n", incoming);
        if (ch == 't' || ch == 'T') {
            Serial.println("[SERIAL] Test command received; starting recording.");
            runMicAndSpeakerTest();
            return;
        }
        // Ignore CR/LF characters sent by Serial Monitor.
    }

    // 3. Continuous microphone monitoring (VAD automatic trigger)
    if (detectSpeechActivity()) {
        Serial.println("\n[VOICE] Continuous VAD: Speech activity detected by microphone!");
        runMicAndSpeakerTest();
        return;
    }

    delay(20);
}
