

#include <ArduinoBLE.h>
#include "LSM6DS3.h"
#include "Wire.h"
#include <Adafruit_LSM6DS3TRC.h> 

// Internal IMU
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

// External IMU
Adafruit_LSM6DS3TRC lsm6ds_6A, lsm6ds_6B; 
Adafruit_Sensor *lsm_temp_6A, *lsm_accel_6A, *lsm_gyro_6A, *lsm_temp_6B, *lsm_accel_6B, *lsm_gyro_6B;


 // Bluetooth® Low Energy Battery Service
BLEService S0_imuService("17649A0-D98E-11E5-9EEC-0002A5D5C51B");

BLEFloatCharacteristic S0_axCharacteristic("917649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S0_ayCharacteristic("917649A2-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S0_azCharacteristic("917649A3-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S0_gxCharacteristic("917649A4-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S0_gyCharacteristic("917649A5-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S0_gzCharacteristic("917649A6-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);

BLEService S1_imuService("17649A1-D98E-11E5-9EEC-0002A5D5C51B");
BLEFloatCharacteristic S1_axCharacteristic("927649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S1_ayCharacteristic("937649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S1_azCharacteristic("947649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S1_gxCharacteristic("957649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S1_gyCharacteristic("967649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S1_gzCharacteristic("977649A1-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);

BLEService S2_imuService("17649A2-D98E-11E5-9EEC-0002A5D5C51B");
BLEFloatCharacteristic S2_axCharacteristic("91764911-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S2_ayCharacteristic("91764921-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S2_azCharacteristic("91764931-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S2_gxCharacteristic("91764941-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S2_gyCharacteristic("91764951-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);
BLEFloatCharacteristic S2_gzCharacteristic("91764961-D98E-11E5-9EEC-0002A5D5C51B", BLERead | BLENotify);



int16_t zero = 0;
long previousMillis = 0; 

void setup() {
  Serial.begin(9600);    // initialize serial communication
//  while (!Serial); 

  // Begin Internal IMU
  if (myIMU.begin() != 0) {
      Serial.println("Failed to find internal LSM6DS");
  } else {
      Serial.println("Device OK!");
  }

  // Begin External IMU
  // External 6A
  if (!lsm6ds_6A.begin_I2C(0x6A)) {
    Serial.println("Failed to find external LSM6DS chip at 0x6A");
    while (1) {
      delay(10);
    }
  }
  Serial.println("LSM6DS Found at 0x6A!");
  lsm_temp_6A = lsm6ds_6A.getTemperatureSensor();
  lsm_temp_6A->printSensorDetails();
  lsm_accel_6A = lsm6ds_6A.getAccelerometerSensor();
  lsm_accel_6A->printSensorDetails();
  lsm_gyro_6A = lsm6ds_6A.getGyroSensor();
  lsm_gyro_6A->printSensorDetails();


  // External 6B
  if (!lsm6ds_6B.begin_I2C(0x6B)) {
    Serial.println("Failed to find external LSM6DS chip at 0x6B");
    while (1) {
      delay(10);
    }
  }
  Serial.println("LSM6DS Found at 0x6B!");
  lsm_temp_6B = lsm6ds_6B.getTemperatureSensor();
  lsm_temp_6B->printSensorDetails();
  lsm_accel_6B = lsm6ds_6B.getAccelerometerSensor();
  lsm_accel_6B->printSensorDetails();
  lsm_gyro_6B = lsm6ds_6B.getGyroSensor();
  lsm_gyro_6B->printSensorDetails();
  

  pinMode(LED_BUILTIN, OUTPUT); // initialize the built-in LED pin to indicate when a central is connected

  // begin initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  
  BLE.setLocalName("SEEED XIAO");
  BLE.setAdvertisedService(S0_imuService); // add the service UUID
  BLE.setAdvertisedService(S1_imuService); 
  BLE.setAdvertisedService(S2_imuService); 

   // add the  characteristic
  S0_imuService.addCharacteristic(S0_axCharacteristic);
  S0_imuService.addCharacteristic(S0_ayCharacteristic);
  S0_imuService.addCharacteristic(S0_azCharacteristic);
  S0_imuService.addCharacteristic(S0_gxCharacteristic); 
  S0_imuService.addCharacteristic(S0_gyCharacteristic);
  S0_imuService.addCharacteristic(S0_gzCharacteristic);

  S1_imuService.addCharacteristic(S1_axCharacteristic);
  S1_imuService.addCharacteristic(S1_ayCharacteristic);
  S1_imuService.addCharacteristic(S1_azCharacteristic);
  S1_imuService.addCharacteristic(S1_gxCharacteristic); 
  S1_imuService.addCharacteristic(S1_gyCharacteristic);
  S1_imuService.addCharacteristic(S1_gzCharacteristic);

  S2_imuService.addCharacteristic(S2_axCharacteristic);
  S2_imuService.addCharacteristic(S2_ayCharacteristic);
  S2_imuService.addCharacteristic(S2_azCharacteristic);
  S2_imuService.addCharacteristic(S2_gxCharacteristic); 
  S2_imuService.addCharacteristic(S2_gyCharacteristic);
  S2_imuService.addCharacteristic(S2_gzCharacteristic);

  // Add the  service
  BLE.addService(S0_imuService); 
  BLE.addService(S1_imuService);
  BLE.addService(S2_imuService);
  
  // start advertising
  BLE.advertise();
  Serial.println("Bluetooth® device active, waiting for connections...");
}



void loop() {
  // wait for a Bluetooth® Low Energy central
  BLEDevice central = BLE.central();

  // if a central is connected to the peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    // print the central's BT address:
    Serial.println(central.address());
    // turn on the LED to indicate the connectbion:
    digitalWrite(LED_BUILTIN, HIGH);

    // check every 200ms
    // while the central is connected:
    while (central.connected()) {
      long currentMillis = millis();
      // if 200ms 
      if (currentMillis - previousMillis >= 200) {
        previousMillis = currentMillis;
        getIMU_Internal();
        getIMU_External(0x6B);
        getIMU_External(0x6A);
      }
    }
    // when the central disconnects, turn off the LED:
    digitalWrite(LED_BUILTIN, LOW);
    Serial.print("Disconnected from central: ");
    Serial.println(central.address());
  }
}


void getIMU_Internal(){
  int16_t ax,ay, az;

  ax = myIMU.readRawAccelX();
  ay = myIMU.readRawAccelY();
  az = myIMU.readRawAccelZ();

  float afx,afy, afz, gfx, gfy, gfz;
  
  afx = myIMU.readFloatAccelX();
  afy = myIMU.readFloatAccelY();
  afz = myIMU.readFloatAccelZ();
  gfx = myIMU.readFloatGyroX();
  gfy = myIMU.readFloatGyroY();
  gfz = myIMU.readFloatGyroZ();
  
  Serial.print("\nS0_Accelerometer:\n");
  Serial.println(afx); 
  Serial.println(afy); 
  Serial.println(afz);
  Serial.print("S0_Gyroscope:\n");
  Serial.println(gfx); 
  Serial.println(gfy);
  Serial.println(gfz);


  S0_axCharacteristic.writeValue(afx);
  S0_ayCharacteristic.writeValue(afy);
  S0_azCharacteristic.writeValue(afz);
  S0_gxCharacteristic.writeValue(gfx);
  S0_gyCharacteristic.writeValue(gfy);
  S0_gzCharacteristic.writeValue(gfz);
  
}


void getIMU_External(uint8_t addr){
  
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;
  
  if(addr == 0x6A){
    lsm_temp_6A->getEvent(&temp);
    lsm_accel_6A->getEvent(&accel);
    lsm_gyro_6A->getEvent(&gyro);
    Serial.print("S2:\n");
  }
  else if(addr == 0x6B){
    lsm_temp_6B->getEvent(&temp);
    lsm_accel_6B->getEvent(&accel);
    lsm_gyro_6B->getEvent(&gyro);
    Serial.print("S1:\n");
  }

  float afx,afy, afz, gfx, gfy, gfz;
  
  afx = accel.acceleration.x;
  afy = accel.acceleration.y;
  afz = accel.acceleration.z;
  gfx = gyro.gyro.x;
  gfy = gyro.gyro.y;
  gfz = gyro.gyro.z;

  // Print Stuff
  Serial.print("Accelerometer:\n");
  Serial.println(afx); 
  Serial.println(afy); 
  Serial.println(afz);
  Serial.print("Gyroscope:\n");
  Serial.println(gfx); 
  Serial.println(gfy);
  Serial.println(gfz);


  if(addr == 0x6A){
    S2_axCharacteristic.writeValue(afx);
    S2_ayCharacteristic.writeValue(afy);
    S2_azCharacteristic.writeValue(afz);
    S2_gxCharacteristic.writeValue(gfx);
    S2_gyCharacteristic.writeValue(gfy);
    S2_gzCharacteristic.writeValue(gfz);
  }
  else if(addr == 0x6B){
    S1_axCharacteristic.writeValue(afx);
    S1_ayCharacteristic.writeValue(afy);
    S1_azCharacteristic.writeValue(afz);
    S1_gxCharacteristic.writeValue(gfx);
    S1_gyCharacteristic.writeValue(gfy);
    S1_gzCharacteristic.writeValue(gfz);
  }
  
 }
