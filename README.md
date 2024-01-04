# audible-leds

Audio Stream
- Synchronous
- Python multiprocessing
- Stream I/O in another thread via a callback
    - Keep running queue of audio samples managed by background thread
    - when samples are queried, fetch from queue
- virtual audio channel that combines multiple sources and reroutes to an audio input (BlackHole)

Signal processing
- Two goals: good audio bins, no "lag" in bins
- Resolution of DFT depends on num samples but increasing the number of samples increases "lag"
- How do we bin?
    - We can linearly group bins
    - We can group logarithmically
        - size 0 bins! Fix with "exponentially growing bins sizes"
        - not a great solution
    - Mel spectrum
How do we smooth our bins?
- convolve with window functions
- exponential decay (this is already kind of achieved with samples passed to dft)
- resolution of dft is even worse since we have linearly spaced bins (in Hz), but the upper range of bins (sampling rate / 2, usually ~22kHz) isn't good for visualization. Human voice + instruments have fundemental frequencies up to about ~1.5kHz, which is only fraction of the range our DFT cover - meaning we have to throw away many bins.

Extra effects:
- Sigmoid to normalize between 0 and 1
    - typically, we normalize dft using the maximum power of a pure tone
    - we can have more power than this, so we get values > 1
- Mirror
- Two channels
- "bass jump" to visualize beat nicely
- dynamic range of sound inspired by our ears
    - marvins room intro was a good example of this

Visualization
- Matplotlib interactive plotting
    - Interactive line plots to visualize spectrum
    - Interactive bar plots for a cool equalizer effect
    - Interactive scatter plot to visualize LEDs

Real LEDs
- Requirements:
    - I wanted to leverage the performance of my macbook so I didn't have to worry about processing speed.
        - Need another device to interface with my macbook
    - wireless! dont want to have to connect my laptop to a device physically to transmit the spectrum data
    - low latency. The visualization does not look good if there is significant latency (>30ms). Especially with beat drops.
    - Don't need super high throughput (with 200LEDs * 3 bytes per LED * 60FPS = 36kB/s)

Bluetooth
- Need separate bluetooth module
- No good latency guarantees, especially with a bluetooth reciever that isn't specialized for low latency

Bluetooth Low Energy (BLE)
- Built into arduino nano ble. Even worse latency than bluetooth. Meant for low energy devices, which didn't matter since I would have a wall connection.
- Arduino nano cannot interface with the LED library I wanted to use (FastLED)

Wifi
- I realized that the latency/throughput of a home network is ridiculously good compared to bluetooth. After all, I download large files and play real-time games over it.
- Did some latency/throughput tests between macbook and windows pc, ~10ms latency when both devices were on WiFi over TCP, ~2ms when desktop was connected via ethernet and laptop was right next to the router. More than enough throughput.

ESP32
- Mircorcontroller with built in wifi capabiltiies and support for FastLED

TCP over home network
- Garbage. I thought that since it didn't have the overhead of Windows/MacOS, the network stack would be very lightweight. Unfortunately, it is also a low power device, so networking performance suffers even with all wifi power savings features turned off. I was getting RTTs of ~40ms on average and sometimes up to ~1s. Unacceptable.

TCP over access point
- Garbage. Throughput was much better, but I think the chip wasn't constantly listening to the tcp stream. I could send LED updates at over 100Hz, but many of the updates would be clumped together, which made the leds look like they were being updated at much lower than 100Hz.
- Also, I would have to connect my laptop to the access point created by the ESP32, which means I had to forfeit my internet connection >:( 

UDP over home network
- UDP is great. Low latency, and packets being sent out of order/being dropped didn't really cause any significant decline in visual fidelity. We could also send time information with packets and drop packets that are sent out of order. I didn't test latecny here since I was so excited when it looked so good visually.

Serial connection to ESP32
- We can also communicate to the ESP32 via a serial connection. My esp32 board supports a baud rate of 460800b/s, or ~58kB/s, which is enough. I didn't do a latency measurement experiement, but anecdotally it was < 3ms.

Hardware:
- LEDs suck a lot of current, so we need a power supply capable of providing high current.

I connected ESP32 to the data pin of a WS2812B LED strip via a 220 ohm resistor.
