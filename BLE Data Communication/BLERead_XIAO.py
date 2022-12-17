import asyncio
import bleak as blk
import warnings
import struct

warnings.simplefilter(action='ignore', category=FutureWarning)


S0_AX_UUID = '917649A1-D98E-11E5-9EEC-0002A5D5C51B'
S0_AY_UUID = '917649A2-D98E-11E5-9EEC-0002A5D5C51B'
S0_AZ_UUID = '917649A3-D98E-11E5-9EEC-0002A5D5C51B'
S0_GX_UUID = '917649A4-D98E-11E5-9EEC-0002A5D5C51B'
S0_GY_UUID = '917649A5-D98E-11E5-9EEC-0002A5D5C51B'
S0_GZ_UUID = '917649A6-D98E-11E5-9EEC-0002A5D5C51B'

S1_AX_UUID = '927649A1-D98E-11E5-9EEC-0002A5D5C51B'
S1_AY_UUID = '937649A1-D98E-11E5-9EEC-0002A5D5C51B'
S1_AZ_UUID = '947649A1-D98E-11E5-9EEC-0002A5D5C51B'
S1_GX_UUID = '957649A1-D98E-11E5-9EEC-0002A5D5C51B'
S1_GY_UUID = '967649A1-D98E-11E5-9EEC-0002A5D5C51B'
S1_GZ_UUID = '977649A1-D98E-11E5-9EEC-0002A5D5C51B'

S2_AX_UUID = '91764911-D98E-11E5-9EEC-0002A5D5C51B'
S2_AY_UUID = '91764921-D98E-11E5-9EEC-0002A5D5C51B'
S2_AZ_UUID = '91764931-D98E-11E5-9EEC-0002A5D5C51B'
S2_GX_UUID = '91764941-D98E-11E5-9EEC-0002A5D5C51B'
S2_GY_UUID = '91764951-D98E-11E5-9EEC-0002A5D5C51B'
S2_GZ_UUID = '91764961-D98E-11E5-9EEC-0002A5D5C51B'

async def main():
    FOUND = False
    dlist = []
    devices = await blk.BleakScanner.discover()
    for d in devices:
        if d.name != None:
            dlist.append(d.name)

        if d.name == 'SEEED XIAO' :
            FOUND = True
            print('Found XIAO!!!')

            async with blk.BleakClient(d.address) as client:
                print(f'Connected to {d.address}')

                services = await client.get_services()

                for service in services:
                    print('service', service.handle, service.uuid, service.description)
                    characteristics = service.characteristics

                    for char in characteristics:
                        print('  characteristic', char.handle, char.uuid, char.description, char.properties)

                        descriptors = char.descriptors

                        for desc in descriptors:
                            print('    descriptor', desc)

                while(True):
                    print("Sensor 0: ")
                    await get_IMU(client, S0_AX_UUID,S0_AY_UUID,S0_AZ_UUID,S0_GX_UUID,S0_GY_UUID,S0_GZ_UUID)
                    print("Sensor 1: ")
                    await get_IMU(client, S1_AX_UUID,S1_AY_UUID,S1_AZ_UUID,S1_GX_UUID,S1_GY_UUID,S1_GZ_UUID)
                    print("Sensor 2: ")
                    await get_IMU(client, S2_AX_UUID,S2_AY_UUID,S2_AZ_UUID,S2_GX_UUID,S2_GY_UUID,S2_GZ_UUID)              


    if not FOUND:
        print("---------  NOT FOUND ----------")
        print("Devices: ", dlist)


async def get_IMU(client, axid,ayid,azid,gxid,gyid,gzid):
    AX_bytes = await client.read_gatt_char(axid)
    AY_bytes = await client.read_gatt_char(ayid)
    AZ_bytes = await client.read_gatt_char(azid)
    GX_bytes = await client.read_gatt_char(gxid)
    GY_bytes = await client.read_gatt_char(gyid)
    GZ_bytes = await client.read_gatt_char(gzid)

    AX = struct.unpack('f', AX_bytes)
    AY = struct.unpack('f', AY_bytes)
    AZ = struct.unpack('f', AZ_bytes)
    GX = struct.unpack('f', GX_bytes)
    GY = struct.unpack('f', GY_bytes)
    GZ = struct.unpack('f', GZ_bytes)


    print("AX", AX[0])
    print("AY", AY[0])
    print("AZ", AZ[0], '\n')
    print("GX", GX[0])
    print("GY", GY[0])
    print("GZ", GZ[0], '\n')

# asyncio.run(main())

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main())
except KeyboardInterrupt:
    print('\nReceived Keyboard Interrupt')
finally:
    print('Program finished')
