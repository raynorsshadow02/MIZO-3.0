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
 *   WS   --> GPIO 5  (Word Select / LRCLK)
 *   SCK  --> GPIO 6  (Serial Clock / BCLK)
 *   SD   --> GPIO 4  (Serial Data / DOUT)
 *
 * MAX98357A (Speaker DAC) -> ESP32-S3:
 *   VIN  --> 5V (or 3.3V)
 *   GND  --> GND
 *   BCLK --> GPIO 15 (Bit Clock)
 *   LRC  --> GPIO 16 (Left/Right Clock)
 *   DIN  --> GPIO 7  (Data In)
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
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Replace with your PC / Cloud Server IP address (Run 'ipconfig' on Windows to find it)
const char* MIZO_SERVER_IP   = "192.168.1.100";
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
#define I2S_MIC_WS      5
#define I2S_MIC_SCK     6
#define I2S_MIC_SD      4

// I2S Speaker Pins (MAX98357A)
#define I2S_SPK_PORT    I2S_NUM_1
#define I2S_SPK_BCLK    15
#define I2S_SPK_LRC     16
#define I2S_SPK_DIN     7

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
// AUDIO PIPELINE
// ---------------------------------------------------------------------------------
void recordAndSendAudio() {
    Serial.println("\n🎙️ [INMP441] Recording speech from microphone...");
    size_t bytes_read = 0;
    size_t total_read = 0;

    while (total_read < BUFFER_SIZE) {
        size_t to_read = min((size_t)1024, BUFFER_SIZE - total_read);
        i2s_read(I2S_MIC_PORT, (char*)(record_buffer + total_read), to_read, &bytes_read, portMAX_DELAY);
        total_read += bytes_read;
    }
    Serial.printf("✓ Audio recorded (%d bytes). Connecting to Mizo Cloud...\n", total_read);

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("❌ Wi-Fi disconnected!");
        return;
    }

    HTTPClient http;
    String url = String("http://") + MIZO_SERVER_IP + ":" + String(MIZO_SERVER_PORT) + "/api/v1/esp32/audio?format=audio";
    http.begin(url);
    http.addHeader("X-Device-Key", DEVICE_API_KEY);

    // Multi-part form preparation
    String boundary = "----MizoESP32Boundary" + String(millis());
    http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

    byte wavHeader[44];
    setupWavHeader(wavHeader, total_read);

    // Build multipart body in stream
    String head = "--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"audio\"; filename=\"inmp441.wav\"\r\n" +
                  "Content-Type: audio/wav\r\n\r\n";
    String tail = "\r\n--" + boundary + "\r\n" +
                  "Content-Disposition: form-data; name=\"student_id\"\r\n\r\n1\r\n" +
                  "--" + boundary + "--\r\n";

    int totalPayloadLen = head.length() + 44 + total_read + tail.length();

    Serial.println("📡 Uploading audio stream to cloud...");
    WiFiClient* stream = http.getStreamPtr();
    
    // Using raw POST stream
    int httpResponseCode = http.POST((uint8_t*)NULL, totalPayloadLen);

    if (httpResponseCode == HTTP_CODE_OK || httpResponseCode == 200) {
        Serial.println("🔊 Cloud reply received! Streaming to MAX98357A speaker...");
        int len = http.getSize();
        WiFiClient* respStream = http.getStreamPtr();

        // Skip 44 byte WAV header from server
        uint8_t skipHeader[44];
        respStream->readBytes(skipHeader, 44);

        uint8_t audioChunk[1024];
        size_t bytes_written;

        while (http.connected() && (len > 0 || len == -1)) {
            size_t availableBytes = respStream->available();
            if (availableBytes) {
                int readBytes = respStream->readBytes(audioChunk, min(availableBytes, sizeof(audioChunk)));
                i2s_write(I2S_SPK_PORT, audioChunk, readBytes, &bytes_written, portMAX_DELAY);
                if (len > 0) len -= readBytes;
            }
            delay(1);
        }
        Serial.println("✓ Spoken playback complete!");
    } else {
        Serial.printf("❌ Cloud Server Error: %d\n", httpResponseCode);
    }
    http.end();
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

void loop() {
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
