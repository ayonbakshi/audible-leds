// #include <WiFi.h>
// #include <WiFiClient.h>

// #include <FastLED.h>

// #include "constants.h"

// #define NUM_LEDS 100      // Number of LEDs in your strip
// #define DATA_PIN 5       // Pin connected to the data line of WS2812B strip
// CRGB leds[NUM_LEDS];
  
// const char* ssid = WIFI_SSID;
// const char* password = WIFI_PASSWORD;

// const char* serverIP = "192.168.0.200";  // Replace with the server's IP address
// const uint16_t serverPort = 8080;       // Replace with the server's port number

// const size_t buffer_size = 1024;
// uint8_t buffer[buffer_size];

// WiFiClient client;

// void setup() {
//   Serial.begin(115200);

//   connectToWiFi();

//   // WiFi.setSleep(WIFI_PS_NONE);

//   FastLED.addLeds<WS2812B, DATA_PIN, GRB>(leds, NUM_LEDS);
//   FastLED.setBrightness(20);
// }

// unsigned long start = 0;
// unsigned long frames = 0;
// void loop() {
//   while (!client.connected()) {
//     connectToServer();
//     client.setNoDelay(true);
//     delay(1000);
//     start = millis();
//     frames = 0;
//   }
  
//   size_t bytesRead = client.readBytesUntil(0xff, buffer, buffer_size);
//   if (bytesRead > 0) {
//     // Serial.println(bytesRead);
//     bufferToLeds(bytesRead);
//     frames++;
//     if (frames % 50 == 0) {
//       unsigned long t = millis();
//       Serial.println(frames * 1000 / (t - start));
//     }
//   }
// }

// void connectToWiFi() {
//   Serial.println("Connecting to WiFi...");
//   WiFi.begin(ssid, password);
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(1000);
//     Serial.println("Connecting...");
//   }
//   Serial.println("Connected to WiFi");
// }

// bool connectToServer() {
//   if (client.connect(serverIP, serverPort)) {
//     Serial.println("Connected to server");
//     return true;
//   } else {
//     Serial.println("Connection to server failed. Retrying...");
//     return false;
//   }
// }

// void bufferToLeds(size_t bytesInBuffer) {
//   for (size_t i = 0; i < bytesInBuffer; i+=3) {
//     leds[i/3] = CRGB(buffer[i], buffer[i+1], buffer[i+2]);
//   }
//   FastLED.show();
// }
