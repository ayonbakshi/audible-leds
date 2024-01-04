#include <WiFi.h>
#include "esp_wifi.h"
#include <WiFiUdp.h>

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
WiFiUDP udp;

unsigned int localUdpPort = 12345;  // local port to listen on

void setup() {
  Serial.begin(460800);
  while (!Serial);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  esp_wifi_set_ps(WIFI_PS_NONE);

  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());

  udp.begin(localUdpPort);
  Serial.printf("%lu Listening on UDP port %d\n", millis(), localUdpPort);

  initFastLED();
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    int len = udp.read(buffer, bufferSize);
    if (len > 0) {
      // Serial.printf("Received %d bytes at %lu\n", len, millis());
      bufferToLeds(len);
    }
  }
}
