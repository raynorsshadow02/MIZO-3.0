/*
 * =================================================================================
 * MIZO 3.0 — ESP32-S3 ROBOTIC TUTOR FIRMWARE
 * Microcontroller: ESP32-S3 (WROOM-1 / N8R2 / N16R8)
 * Audio In:  INMP441 I2S MEMS Microphone
 * Audio Out: MAX98357A I2S Class-D DAC Amplifier + Speaker
 * =================================================================================
 * 
 * HARDWARE WIRING:
 * ---------------------------------------------------------------------------------
 * INMP441 (Microphone) -> ESP32-S3:
 *   VDD  --> 3.3V
 *   GND  --> GND
 *   L/R  --> GND (Left Channel)
 *   SCK  --> GPIO 4  (Serial Clock / BCLK)
 *   WS   --> GPIO 5  (Word Select / LRCLK)
 *   SD   --> GPIO 6  (Serial Data / DOUT)
 *
 * MAX98357A (Speaker DAC) -> ESP32-S3:
 *   VIN  --> 5V (or 3.3V)
 *   GND  --> GND
 *   BCLK --> GPIO 18 (Bit Clock)
 *   LRC  --> GPIO 21 (Left/Right Clock / WS)
 *   DIN  --> GPIO 20 (Data In)
 *   GAIN --> GND (9dB gain)
 *
 * PUSH-TO-TALK BUTTON (Optional):
 *   Button Pin --> GPIO 0 (BOOT button on ESP32-S3)
 * =================================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>

// ---------------------------------------------------------------------------------
// CONFIGURATION (Update with your Wi-Fi and Server IP)
// ---------------------------------------------------------------------------------
const char* WIFI_SSID     = "SHAHID-4G";
const char* WIFI_PASSWORD = "kingshaikh";

// Replace with your PC / Cloud Server IP address (Run 'ipconfig' on Windows to find it)
const char* MIZO_SERVER_IP   = "192.168.1.192";
const int   MIZO_SERVER_PORT = 8000;

// Device API key (Configured in Mizo Admin Dashboard)
const char* DEVICE_API_KEY   = "mizo_esp32_secret_key_123";

// Audio Parameters
#define SAMPLE_RATE     16000
#define BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_16BIT
#define RECORD_SECONDS  5
#define BUFFER_SIZE     (SAMPLE_RATE * 2 * RECORD_SECONDS) // 16-bit = 2 bytes per sample

// I2S Microphone Pins (INMP441)
#define I2S_MIC_PORT    I2S_NUM_0
#define I2S_MIC_SCK     4  // Serial Clock (BCLK)
#define I2S_MIC_WS      5  // Word Select (LRCLK)
#define I2S_MIC_SD      6  // Serial Data (DOUT)

// I2S Speaker Pins (MAX98357A)
#define I2S_SPK_PORT    I2S_NUM_1
#define I2S_SPK_BCLK    18 // Bit Clock (BCLK)
#define I2S_SPK_LRC     21 // Left/Right Clock (LRCLK / WS)
#define I2S_SPK_DIN     20 // Data In (DIN)

// Push-to-talk button
#define BUTTON_PIN      0

// Audio recording buffer
uint8_t* record_buffer = NULL;

// ---------------------------------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------------------------------
void setupI2SMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
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
        .bits_per_sample = BITS_PER_SAMPLE,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
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
    header[20] = 1; header[21] = 0; // PCM
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
    header[34] = 16; header[35] = 0; // 16 bits
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    header[40] = (byte)(totalAudioLen & 0xff);
    header[41] = (byte)((totalAudioLen >> 8) & 0xff);
    header[42] = (byte)((totalAudioLen >> 16) & 0xff);
    header[43] = (byte)((totalAudioLen >> 24) & 0xff);
}

// ---------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------
// SPEAKER HARDWARE AUDIO TEST
// ---------------------------------------------------------------------------------
void testSpeaker() {
    Serial.println("\n🔊 [MAX98357A] Running speaker hardware audio self-test...");
    const int freq = 440; // 440Hz test tone
    const int num_samples = SAMPLE_RATE / 2; // 0.5s duration
    int16_t sample_buffer[256];
    size_t bytes_written = 0;

    int sample_idx = 0;
    while (sample_idx < num_samples) {
        int chunk_samples = min(256, num_samples - sample_idx);
        for (int i = 0; i < chunk_samples; i++) {
            float t = (float)(sample_idx + i) / SAMPLE_RATE;
            sample_buffer[i] = (int16_t)(sin(2 * 3.14159265 * freq * t) * 6000); // 16-bit PCM amplitude
        }
        i2s_write(I2S_SPK_PORT, sample_buffer, chunk_samples * sizeof(int16_t), &bytes_written, portMAX_DELAY);
        sample_idx += chunk_samples;
    }
    Serial.println("✓ [MAX98357A] Speaker hardware self-test complete.");
}

// ---------------------------------------------------------------------------------
// AUDIO PIPELINE (INMP441 Mic In -> Cloud -> MAX98357A Speaker Out)
// ---------------------------------------------------------------------------------
void recordAndSendAudio() {
    Serial.println("\n🎙️ [INMP441] Recording speech from external microphone...");
    size_t bytes_read = 0;
    size_t total_read = 0;

    while (total_read < BUFFER_SIZE) {
        size_t to_read = min((size_t)1024, BUFFER_SIZE - total_read);
        i2s_read(I2S_MIC_PORT, (char*)(record_buffer + total_read), to_read, &bytes_read, portMAX_DELAY);
        total_read += bytes_read;
    }
    Serial.printf("✓ Audio recorded (%d bytes). Connecting to Mizo Cloud (%s:%d)...\n", total_read, MIZO_SERVER_IP, MIZO_SERVER_PORT);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("❌ Wi-Fi disconnected!");
        return;
    }

    WiFiClient client;
    if (!client.connect(MIZO_SERVER_IP, MIZO_SERVER_PORT)) {
        Serial.printf("❌ Could not connect to Mizo Cloud Server at %s:%d\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
        return;
    }

    // Multipart form-data preparation
    String boundary = "----MizoESP32Boundary" + String(millis());
    byte wavHeader[44];
    setupWavHeader(wavHeader, total_read);

    String head = "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"audio\"; filename=\"inmp441.wav\"\r\n" +
                  "Content-Type: audio/wav\r\n\r\n";
    String tail = "\r\n--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"student_id\"\r\n\r\n1\r\n" +
                  "--" + boundary + "--\r\n";

    int totalPayloadLen = head.length() + 44 + total_read + tail.length();

    Serial.println("📡 Uploading audio stream to /api/v1/esp32/audio?format=audio...");
    client.print("POST /api/v1/esp32/audio?format=audio HTTP/1.1\r\n");
    client.printf("Host: %s:%d\r\n", MIZO_SERVER_IP, MIZO_SERVER_PORT);
    client.printf("X-Device-Key: %s\r\n", DEVICE_API_KEY);
    client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
    client.printf("Content-Length: %d\r\n", totalPayloadLen);
    client.print("Connection: close\r\n\r\n");

    // 1. Send multipart prefix
    client.print(head);

    // 2. Send 44-byte WAV header
    client.write(wavHeader, 44);

    // 3. Send audio payload in 1024-byte chunks
    size_t offset = 0;
    while (offset < total_read) {
        size_t chunk = (total_read - offset > 1024) ? 1024 : (total_read - offset);
        client.write(record_buffer + offset, chunk);
        offset += chunk;
    }

    // 4. Send multipart suffix
    client.print(tail);
    client.flush();
    Serial.println("✓ Audio uploaded! Waiting for Mizo voice reply...");

    // Wait for response headers with 30s timeout
    unsigned long timeout = millis() + 30000;
    while (client.connected() && !client.available()) {
        if (millis() > timeout) {
            Serial.println("❌ Timeout waiting for server audio response.");
            client.stop();
            return;
        }
        delay(10);
    }

    // Read HTTP response status line
    String statusLine = client.readStringUntil('\n');
    statusLine.trim();
    Serial.printf("Server response: %s\n", statusLine.c_str());

    if (!statusLine.startsWith("HTTP/1.1 200") && !statusLine.startsWith("HTTP/1.0 200")) {
        Serial.printf("❌ Server error response: %s\n", statusLine.c_str());
        client.stop();
        return;
    }

    // Skip HTTP headers
    bool inHeader = true;
    while (client.connected() && inHeader) {
        String line = client.readStringUntil('\n');
        line.trim();
        if (line.length() == 0) {
            inHeader = false;
            break;
        }
    }

    // Stream incoming WAV audio directly to MAX98357A I2S DAC speaker
    Serial.println("🔊 Cloud reply received! Streaming to MAX98357A speaker...");
    uint8_t skipHeader[44];
    size_t headerRead = 0;
    while (headerRead < 44 && client.connected()) {
        if (client.available()) {
            skipHeader[headerRead++] = client.read();
        } else {
            delay(1);
        }
    }

    uint8_t audioChunk[1024];
    size_t bytes_written = 0;
    while (client.connected() || client.available()) {
        size_t avail = client.available();
        if (avail > 0) {
            int toRead = min(avail, sizeof(audioChunk));
            int readBytes = client.readBytes(audioChunk, toRead);
            if (readBytes > 0) {
                i2s_write(I2S_SPK_PORT, audioChunk, readBytes, &bytes_written, portMAX_DELAY);
            }
        } else {
            delay(1);
        }
    }
    client.stop();
    Serial.println("✓ Spoken playback complete!");
}

// ---------------------------------------------------------------------------------
// ARDUINO SETUP & LOOP
// ---------------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=============================================");
    Serial.println("   MIZO 3.0 ESP32-S3 ROBOTIC TUTOR CLIENT    ");
    Serial.println("=============================================");

    pinMode(BUTTON_PIN, INPUT_PULLUP);

    // Allocate PSRAM or Heap buffer
    record_buffer = (uint8_t*)ps_malloc(BUFFER_SIZE);
    if (!record_buffer) {
        record_buffer = (uint8_t*)malloc(BUFFER_SIZE);
    }
    if (!record_buffer) {
        Serial.println("❌ Failed to allocate recording buffer!");
        while (1) delay(1000);
    }

    setupI2SMicrophone();
    setupI2SSpeaker();

    // Run working speaker test on startup
    testSpeaker();

    // Connect to Wi-Fi
    Serial.printf("Connecting to Wi-Fi '%s'...", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\n✓ Wi-Fi Connected! IP: %s\n", WiFi.localIP().toString().c_str());
    Serial.println("Ready! Continuous microphone monitoring ACTIVE in SLEEP.");
    Serial.println("Say 'Mizo' anytime to wake up. (Button optional for testing).");
}

// ---------------------------------------------------------------------------------
// CONTINUOUS MICROPHONE MONITORING (BUTTONLESS VAD)
// ---------------------------------------------------------------------------------
#define VAD_SAMPLE_COUNT 512
#define SPEECH_THRESHOLD 1200 // Audio amplitude threshold for speech detection

bool detectSpeechActivity() {
    int16_t samples[VAD_SAMPLE_COUNT];
    size_t bytes_read = 0;
    i2s_read(I2S_MIC_PORT, (char*)samples, sizeof(samples), &bytes_read, 30 / portTICK_PERIOD_MS);
    if (bytes_read == 0) return false;

    int num_samples = bytes_read / sizeof(int16_t);
    int32_t total_amplitude = 0;
    for (int i = 0; i < num_samples; i++) {
        total_amplitude += abs((int32_t)samples[i]);
    }
    int32_t avg_amplitude = total_amplitude / num_samples;
    return (avg_amplitude > SPEECH_THRESHOLD);
}

unsigned long lastHeartbeat = 0;

void checkAndSendHeartbeat() {
    if (millis() - lastHeartbeat >= 20000) {
        lastHeartbeat = millis();
        if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            String url = String("http://") + MIZO_SERVER_IP + ":" + String(MIZO_SERVER_PORT) + "/api/v1/esp32/ping";
            http.begin(url);
            http.addHeader("X-Device-Key", DEVICE_API_KEY);
            int code = http.GET();
            http.end();
        }
    }
}

void loop() {
    // 0. Heartbeat to server so dashboard knows hardware is online
    checkAndSendHeartbeat();

    // 1. Continuous Microphone Monitoring: user speaks -> audio captured automatically
    // The listener never completely shuts down. Button is NOT required.
    if (detectSpeechActivity()) {
        Serial.println("\n[VOICE] Speech activity detected by microphone!");
        recordAndSendAudio();
        delay(400); // Cooldown after audio playback before resuming standby monitoring
        return;
    }

    // 2. Optional Manual Button (Retained for development/testing, NOT required)
    if (digitalRead(BUTTON_PIN) == LOW) {
        delay(50); // Debounce
        if (digitalRead(BUTTON_PIN) == LOW) {
            Serial.println("\n🔘 [Manual Dev Trigger] Button pressed. Recording audio...");
            recordAndSendAudio();
            delay(1000); // Prevent double trigger
        }
    }

    delay(25);
}

