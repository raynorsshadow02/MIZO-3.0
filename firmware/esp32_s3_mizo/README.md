# Mizo 3.0 — ESP32-S3 Hardware Client Guide

This folder contains the complete Arduino C++ firmware for connecting an **ESP32-S3 microcontroller** to the **Mizo 3.0 Cloud Server** using an **INMP441 I2S MEMS microphone** and a **MAX98357A I2S DAC amplifier + speaker**.

---

## 1. Hardware Bill of Materials (BOM)

| Component | Description | Interface |
| :--- | :--- | :--- |
| **ESP32-S3 DevKit** | Dual-core Xtensa 240MHz, Wi-Fi 4 + BLE 5, 8MB Flash, 2MB/8MB PSRAM | Microcontroller |
| **INMP441** | High-precision omnidirectional MEMS microphone | I2S Digital In |
| **MAX98357A** | 3.2W mono Class-D audio amplifier | I2S Digital Out |
| **Speaker** | 4Ω or 8Ω, 3W dynamic speaker | Analog Output |
| **Jumper Wires & Breadboard** | Standard prototyping cables | — |

---

## 2. Wiring Diagram

### INMP441 Microphone ➔ ESP32-S3
| INMP441 Pin | ESP32-S3 Pin | Function |
| :--- | :--- | :--- |
| **VDD** | `3.3V` | Power (3.3V only) |
| **GND** | `GND` | Ground |
| **L/R** | `GND` | Left channel select |
| **SCK** | `GPIO 4` | Serial Clock (BCLK) |
| **WS** | `GPIO 5` | Word Select (LRCLK) |
| **SD** | `GPIO 6` | Serial Data Output (DOUT) |

### MAX98357A Amplifier ➔ ESP32-S3
| MAX98357A Pin | ESP32-S3 Pin | Function |
| :--- | :--- | :--- |
| **VIN** | `5V` (or `3.3V`) | Power (5V provides louder 3W audio) |
| **GND** | `GND` | Ground |
| **BCLK** | `GPIO 18` | Bit Clock |
| **LRC** | `GPIO 21` | Left / Right Clock (WS) |
| **DIN** | `GPIO 20` | Digital Audio Data Input |
| **GAIN** | `GND` | 9dB Gain setting (leave open for 12dB) |
| **SD_MODE** | *Unconnected* | Stereo downmix to mono |

### Speaker ➔ MAX98357A
- Connect Speaker `+` to MAX98357A `+` terminal.
- Connect Speaker `-` to MAX98357A `-` terminal.

---

## 3. Flashing with Arduino IDE

1. **Install ESP32 Board Package**:
   - Open **Arduino IDE** ➔ **File** ➔ **Preferences**.
   - Add this URL to *Additional Boards Manager URLs*:
     ```
     https://espressif.github.io/arduino-esp32/package_esp32_index.json
     ```
   - Go to **Tools** ➔ **Board** ➔ **Boards Manager**, search for `esp32` by Espressif Systems and install it.

2. **Select Board Settings**:
   - **Board**: `ESP32S3 Dev Module`
   - **USB CDC On Boot**: `Enabled`
   - **Flash Size**: `8MB` or `16MB` (match your module)
   - **PSRAM**: `OPI PSRAM` or `QSPI PSRAM` (Enabled)
   - **Upload Speed**: `921600`
   - **Port**: Select the COM port corresponding to your ESP32-S3.

3. **Configure Firmware**:
   - Open [esp32_s3_mizo.ino](file:///c:/Users/samee/Desktop/mizo%203.0/firmware/esp32_s3_mizo/esp32_s3_mizo.ino).
   - Change `WIFI_SSID` and `WIFI_PASSWORD` to your local Wi-Fi credentials.
   - Set `MIZO_SERVER_IP` to your computer's local IP (Run `ipconfig` in Command Prompt to find your IPv4 address, e.g., `192.168.1.100`).
   - Leave `MIZO_SERVER_PORT` as `8000`.
   - Click **Upload**.

---

## 4. Exclusive Hardware Audio Architecture

Mizo 3.0 uses the ESP32-S3 as the **exclusive audio device**:
- **Audio Input**: Captured exclusively by the **INMP441 I2S microphone** (laptop microphone capture disabled).
- **Audio Output**: Played exclusively through the **MAX98357A I2S amplifier and external speaker** (laptop speaker playback disabled).
- **Web Dashboard**: Kept for monitoring session progress, metrics, and text chat, but does not capture microphone audio or play voice audio through laptop speakers.
- **Hardware Disconnect Protection**: If the ESP32-S3 is disconnected or offline, the dashboard displays a clear status message. No automatic fallback to the laptop microphone or speakers occurs.

---

## 5. How It Works

1. **Boot & Speaker Hardware Self-Test**:
   - On startup, the firmware runs `testSpeaker()` playing a 440Hz test sine tone for 500ms through `GPIO 7 (DIN), GPIO 15 (BCLK), GPIO 16 (LRC)` into the MAX98357A to immediately verify audio output and speaker wiring.
2. **Heartbeat Monitoring**:
   - The ESP32 periodically pings `GET /api/v1/esp32/ping` so the server and web dashboard recognize the hardware as online in real time.
3. **Continuous Voice & Push-to-Talk**:
   - The **INMP441** microphone monitors speech activity automatically (or press the **BOOT button (GPIO 0)**).
   - Audio is captured at 16kHz 16-bit Mono.
4. **Cloud Voice Pipeline**:
   - The board POSTs the recorded WAV stream to `POST /api/v1/esp32/audio?format=audio`.
   - The server transcribes speech with Whisper STT, routes through Mizo agents (Tutor, Coach, Speech, Normal Chat), manages memory & onboarding, and synthesizes 16kHz WAV audio.
5. **Direct Hardware Audio Playback**:
   - The binary WAV response stream from the server is streamed directly into the **MAX98357A** I2S DAC, playing Mizo's voice response through the external speaker.

