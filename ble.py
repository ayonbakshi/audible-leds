import asyncio
from bleak import BleakScanner, BleakClient
import time

async def main():
    devices = await BleakScanner.discover(timeout=1)  # Scan for devices for 10 seconds

    # Print out discovered devices
    address = None
    for device in devices:
        # print(f"Device name: {device.name}, Address: {device.address}")
        if "Ayon" in str(device.name) and "IPhone" in str(device.name):
            address = device.address
            break
    
    if address is None:
        raise ValueError("what")
    
    print(f"{address=}")
    
    char_uuid = "00002a11-0000-1000-8000-00805f9b34fb"

    async with BleakClient(address) as client:
        # Connect to the BLE peripheral
        await client.connect()

        bytes_per_message = 100
        messages = 1

        # Define the data you want to send
        # data_to_send = bytearray([1]*bytes_per_message)
        data_to_send = b"ur stupid 2"

        # Write data to a specific characteristic on the peripheral
        start = time.time()
        cnt = 1
        for _ in range(messages):
            await client.write_gatt_char(char_uuid, data_to_send)
            cnt += 1
        total_time = time.time() - start
        avg_time = total_time/cnt

        print(f"fps {1/avg_time}")
        print(f"avg time {avg_time}s")
        print(f"throughput (bps) {(8 * bytes_per_message * messages) / total_time}")

        # Disconnect from the peripheral
        await client.disconnect()


# Run the discovery coroutine
asyncio.run(main())