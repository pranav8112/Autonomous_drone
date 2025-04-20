import cv2
import numpy as np
import pandas as pd
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time
import argparse
import os
from ultralytics import YOLO
from picamera2 import Picamera2, Preview
import cvzone
import datetime  # Import for handling date and time

# Initialize the detection model (1OB model)
model = YOLO('/home/senorita/Downloads/google-coral-usb-raspberry-pi4-main/1OB/best_full_integer_quant_edgetpu.tflite', task='detect')

# Set up argument parser to allow input of connection string
parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='127.0.0.1:14550')
args = parser.parse_args()

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % args.connect)
vehicle = connect(args.connect, baud=921600, wait_ready=True)

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.configure("preview")

# Start the camera
picam2.start()

# Frame dimensions and frame center for target offset calculation
frame_width = 640
frame_height = 480
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2

# Read class names from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Create directory for saving outputs
output_dir = "Aerothon"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize video writer (to save the video in .avi format)
video_filename = os.path.join(output_dir, "output.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

# Initialize log file for hotspot detection timestamps
hotspot_log_filename = os.path.join(output_dir, "hotspot_log.txt")

# Initialize variables for counting and snapshot for Hotspot
hotspot_count = 0  # Count of unique "Hotspot" objects detected
hotspot_last_detected = None  # Timestamp of last "Hotspot" detection
cooldown_duration = 5  # Cooldown duration (5 seconds) to avoid counting the same "Hotspot" repeatedly
snapshot_delay = 0.5  # Delay of 0.5 seconds before taking a snapshot


prev_time = time.time()

# Function to log hotspot detection timestamp
def log_hotspot_detection():
    with open(hotspot_log_filename, "a") as hotspot_log_file:
        detection_time = datetime.datetime.now()
        log_entry = f"Hotspot detected at {detection_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        hotspot_log_file.write(log_entry)
        print(log_entry)  # Optionally print to console for real-time feedback

# Function to arm and take off to a specified altitude
def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

# Function to send body-frame velocity commands to the drone
def send_body_velocity(vx, vy, vz):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0, mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0, 0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# Function to align the drone based on x, y offsets
def align_drone(x_offset, y_offset, offset_threshold=20, velocity=0.4):
    aligned = False
    if abs(x_offset) > offset_threshold:
        vy = velocity if x_offset > 0 else -velocity
        send_body_velocity(0, vy, 0)
        time.sleep(0.5)
    elif abs(y_offset) > offset_threshold:
        vx = -velocity if y_offset > 0 else velocity
        send_body_velocity(vx, 0, 0)
        time.sleep(0.5)
    else:
        aligned = True
    send_body_velocity(0, 0, 0)
    return aligned

# Function to perform payload drop
def drop_parcel():
    msg = vehicle.message_factory.command_long_encode(
        0, 0, mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        7,  # Servo number
        2000,  # Position
        0, 0, 0, 0, 0
    )
    print("Dropping PAYLOAD...")
    vehicle.send_mavlink(msg)
    print("PAYLOAD dropped.")

def main():
    target_altitude = 6
    arm_and_takeoff(target_altitude)
    print("Takeoff complete. Starting mission...")
    vehicle.mode = VehicleMode("AUTO")
    while not vehicle.mode.name == 'AUTO':
        print("Waiting for drone to enter AUTO mode...")
        time.sleep(1)

    frame_center = (320, 240)
    payload_dropped = False  # Flag to track if payload has been dropped

    while True:
        frame = picam2.capture_array()

        # Predict using the YOLO model (detect both Hotspot and Target)
        results = model.predict(frame, imgsz=640)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        # Draw bounding boxes and labels for detections
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            confidence = row[4]
            d = int(row[5])
            c = class_list[d]

            label = f'{c} {confidence:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, label, (x1, y1), 1, 1)

            # Hotspot detection logic
            if c == "Hotspot":
                curr_time = time.time()

                if hotspot_last_detected is None or (curr_time - hotspot_last_detected >= cooldown_duration):
                    hotspot_count += 1
                    hotspot_last_detected = curr_time

                    # Log the detection timestamp
                    log_hotspot_detection()

                    # Delay before taking a snapshot
                    time.sleep(snapshot_delay)
                    hotspot_snapshot = frame[y1:y2, x1:x2]
                    snapshot_filename = os.path.join(output_dir, f"hotspot_{hotspot_count}.jpg")
                    cv2.imwrite(snapshot_filename, hotspot_snapshot)

            elif c == "Target" and not payload_dropped:
                target_center_x = (x1 + x2) // 2
                target_center_y = (y1 + y2) // 2

                x_offset = target_center_x - frame_center_x
                y_offset = target_center_y - frame_center_y

                cv2.circle(frame, (target_center_x, target_center_y), 3, (0, 255, 255), -1)
                cv2.circle(frame, (frame_center_x, frame_center_y), 3, (255, 0, 0), -1)
                cv2.line(frame, (frame_center_x, frame_center_y), (target_center_x, target_center_y), (0, 0, 255), 2)
                cv2.putText(frame, f"X offset: {x_offset}, Y offset: {y_offset}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                vehicle.mode = VehicleMode("GUIDED")
                while vehicle.mode.name != 'GUIDED':
                    time.sleep(1)

                if align_drone(x_offset, y_offset):
                    print("Initial alignment complete. Hovering over target...")

                    target_altitude = 4
                    while vehicle.location.global_relative_frame.alt > target_altitude:
                        send_body_velocity(0, 0, 0.3)
                        time.sleep(0.5)

                    send_body_velocity(0, 0, 0)
                    time.sleep(5)

                    drop_parcel()
                    payload_dropped = True

                vehicle.mode = VehicleMode("AUTO")
                while vehicle.mode.name != 'AUTO':
                    time.sleep(1)

        cv2.imshow("Drone Camera Feed", frame)
        video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    cv2.destroyAllWindows()
    vehicle.close()
    print("Mission complete.")

if __name__ == "__main__":
    main()