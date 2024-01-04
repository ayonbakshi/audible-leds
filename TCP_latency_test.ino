#include <WiFi.h>

#include <FastLED.h>

#define NUM_LEDS 100      // Number of LEDs in your strip
#define DATA_PIN 5       // Pin connected to the data line of WS2812B strip
CRGB leds[NUM_LEDS];

// const char* ssid = "ESP32-Access-Point"; // Access point SSID
// const char* password = "password123";    // Access point password
// const int serverPort = 12345;            // Port to listen on

// WiFiServer server(serverPort);
// WiFiClient client;

const size_t bufferSize = 1024;
uint8_t buffer[bufferSize];

// void setup() {
//   Serial.begin(460800);
//   while (!Serial);       // Wait for serial port to connect

//   // initWiFi();
//   // initFastLED();
// }

// void initWiFi() {
//   // Create an access point
//   WiFi.softAP(ssid, password);
  
//   Serial.println("Access Point created");
  
//   // Start the server
//   server.begin();
  
//   Serial.print("Server started on port ");
//   Serial.println(serverPort);
// }

// unsigned long last = 0;
// void loop() {
//   // if (client.connected()) {
//   //   size_t bytesRead = readFromStream(client);
//   //   bufferToLeds(bytesRead);
//   // } else {
//   //   client = server.available();
//   // }

//   size_t bytesRead = readBytesSerial(0xff);
//   Serial.print(millis()-last);
//   Serial.print(" ");
//   Serial.println(bytesRead);
//   last = millis();
//   bufferToLeds(bytesRead);
// }

// size_t readBytesSerial(char terminator) {
//   size_t i = 0;
//   while (true) {
//     while (Serial.available()) {
//       char c = Serial.read();
//       if (c == terminator) return i;
//       buffer[i] = c;
//       ++i;
//     }
//   }
// }

// unsigned long start = 0;
// unsigned long frames = 0;
// size_t readFromStream(Stream &s) {
//   if (start == 0) start = millis();

//   if (s.available()) {
//     size_t bytesRead = s.readBytesUntil(0xff, buffer, bufferSize);
//     // String data = client.readStringUntil('\n');
//     // String data = String((char *) buffer, bytesRead);
//     // Serial.println(data);
//     // client.print(String(millis()));
//     if (bytesRead > 0) {
//       // Serial.println(bytesRead);
//       frames++;
//       if (frames % 50 == 0) {
//         unsigned long t = millis();
//         Serial.println(frames * 1000 / (t - start));
//         Serial.println(bytesRead);
//       }
//     }
//   }
// }

void initFastLED() {
  FastLED.addLeds<WS2812B, DATA_PIN, GRB>(leds, NUM_LEDS);
  FastLED.setBrightness(20);
}

void bufferToLeds(size_t bytesInBuffer) {
  for (size_t i = 0; i < bytesInBuffer; i+=3) {
    leds[i/3] = CRGB(buffer[i], buffer[i+1], buffer[i+2]);
  }
  FastLED.show();
}
