import smbus
import math
import time

BMI160_I2C_ADDR = 0x69
ACCEL_SENSITIVITY = 16384.0

bus = smbus.SMBus(1)


def write_register(addr, reg, data):
    bus.write_byte_data(addr, reg, data)


def read_register(addr, reg, length):
    return bus.read_i2c_block_data(addr, reg, length)


def initialize_bmi160():
    write_register(BMI160_I2C_ADDR, 0x7E, 0x11)  # ACC_NORMAL_MODE
    write_register(BMI160_I2C_ADDR, 0x7E, 0x15)  # GYR_NORMAL_MODE
    time.sleep(0.1)


def sensor_status():
    status = read_register(BMI160_I2C_ADDR, 0x03, 1)[0]
    acc_mode = (status >> 4) & 0x03
    gyro_mode = (status >> 2) & 0x03
    return status, acc_mode, gyro_mode


def sensor_on(verbose=True):
    write_register(BMI160_I2C_ADDR, 0x7E, 0x11)  # ACC_NORMAL_MODE
    time.sleep(0.1)
    write_register(BMI160_I2C_ADDR, 0x7E, 0x15)  # GYR_NORMAL_MODE
    time.sleep(0.1)

    if verbose:
        status, acc_mode, gyro_mode = sensor_status()
        #print("Sensor ON | status:", status, "| ACC mode:", acc_mode, "| GYRO mode:", gyro_mode)


def sensor_sleep(verbose=True):
    write_register(BMI160_I2C_ADDR, 0x7E, 0x10)  # ACC_SUSPEND
    time.sleep(0.1)
    write_register(BMI160_I2C_ADDR, 0x7E, 0x14)  # GYR_SUSPEND
    time.sleep(0.1)

    if verbose:
        status, acc_mode, gyro_mode = sensor_status()
        #print("Sensor SLEEP | status:", status, "| ACC mode:", acc_mode, "| GYRO mode:", gyro_mode)


def read_raw_acceleration():
    data = read_register(BMI160_I2C_ADDR, 0x12, 6)
    ax_raw = int.from_bytes(data[0:2], "little") - (1 << 16 if data[1] & 0x80 else 0)
    ay_raw = int.from_bytes(data[2:4], "little") - (1 << 16 if data[3] & 0x80 else 0)
    az_raw = int.from_bytes(data[4:6], "little") - (1 << 16 if data[5] & 0x80 else 0)
    return ax_raw, ay_raw, az_raw


def read_raw_gyroscope():
    data = read_register(BMI160_I2C_ADDR, 0x0C, 6)
    gx_raw = int.from_bytes(data[0:2], "little") - (1 << 16 if data[1] & 0x80 else 0)
    gy_raw = int.from_bytes(data[2:4], "little") - (1 << 16 if data[3] & 0x80 else 0)
    gz_raw = int.from_bytes(data[4:6], "little") - (1 << 16 if data[5] & 0x80 else 0)
    return gx_raw, gy_raw, gz_raw


def auto_calibrate():
    print("Starting auto-calibration...")
    num_samples = 100
    ax_offset = 0
    ay_offset = 0
    az_offset = 0

    for _ in range(num_samples):
        ax_raw, ay_raw, az_raw = read_raw_acceleration()
        ax_offset += ax_raw
        ay_offset += ay_raw
        az_offset += az_raw
        time.sleep(0.01)

    ax_offset //= num_samples
    ay_offset //= num_samples
    az_offset //= num_samples

    az_offset -= int(ACCEL_SENSITIVITY)

    print("Auto-calibration completed.")
    print(f"Offsets - X: {ax_offset}, Y: {ay_offset}, Z: {az_offset}")

    return ax_offset, ay_offset, az_offset


def read_acceleration(ax_offset=0, ay_offset=0, az_offset=0):
    ax_raw, ay_raw, az_raw = read_raw_acceleration()
    ax = ((ax_raw - ax_offset) / ACCEL_SENSITIVITY) * 9.81
    ay = ((ay_raw - ay_offset) / ACCEL_SENSITIVITY) * 9.81
    az = ((az_raw - az_offset) / ACCEL_SENSITIVITY) * 9.81
    return ax, ay, az


def read_gyroscope():
    return read_raw_gyroscope()


def calculate_tilt_angles(ax, ay, az):
    pitch = math.atan2(ay, math.sqrt(ax ** 2 + az ** 2)) * 180.0 / math.pi
    roll = math.atan2(-ax, az) * 180.0 / math.pi
    return pitch, roll
