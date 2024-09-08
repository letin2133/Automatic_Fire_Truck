
import RPi.GPIO as GPIO
import time      
from time import sleep
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO



in1 = 13
in2 = 12
in3 = 21
in4 = 20
en1 = 6
en2 = 26
temp1=1

def forward():
    setSpeedDefault(40)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    time.sleep(0.5)
    stop()

def backward():
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    time.sleep(0.5)
    stop()

def turnright():
    setSpeedDefault(30)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.HIGH)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW)
    time.sleep(0.1)
    stop()

def turnleft():
    setSpeedDefault(30)
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.HIGH)
    GPIO.output(in4,GPIO.LOW)
    time.sleep(0.1)
    stop()

def stop():
    GPIO.output(in1,GPIO.LOW)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.LOW) 

def setSpeedDefault(speed):
    p.start(speed)
    q.start(speed)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setup(en1,GPIO.OUT)
GPIO.setup(en2,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)
p=GPIO.PWM(en1,700)
q=GPIO.PWM(en2,700)

p.start(40)
q.start(40)









class_name = ['FIRE','NON-FIRE']
fire_detector = cv2.CascadeClassifier('/home/pc/Desktop/Project/fire_cascade.xml')

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="/home/pc/Desktop/Project/Model/Recognition_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



cam = cv2.VideoCapture(0)
# Set the GPIO mode (BCM or BOARD)
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin controls the pump via the relay module chan pin 18
RELAY_PIN = 18

# Set the relay pin as an output pin
GPIO.setup(RELAY_PIN, GPIO.OUT)

fire_detected = False

threshold_large_fire = 20000


while True:
    is_connect,im = cam.read()
    if (is_connect):
        frame_center_x = im.shape[1] // 2
        frame_center_y = im.shape[0] // 2

        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fires = fire_detector.detectMultiScale(grey, 1.3, 5)
        for x, y, w, h in fires:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = cv2.resize(im[y+3:y+h-3,x+3:x+w-3], (28,28))
            
            # Preprocess input data to match the input tensor shape and type.
            
            input_data = np.expand_dims(roi, axis=0)  # Add batch dimension
            input_data = input_data.astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the output tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            result = np.argmax(output_data)

            
            cv2.putText(im, class_name[result], (x+15, y-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)
            if class_name[result] == 'FIRE' and not fire_detected:
                fire_detected = True

                fire_center_x = x + (w // 2)
                fire_center_y = y + (h // 2)
                
                
                fire_size = w * h
                if fire_size > threshold_large_fire:
                    print("Large fire detected")
                    
                # Perform action for large fire (e.g., call firefighters)

                # Turn the relay ON (HIGH) to turn on the pump
                    GPIO.output(RELAY_PIN, GPIO.HIGH)
                    time.sleep(3)  # Wait for 3 seconds

                # Turn the relay OFF (LOW) to turn off the pump
                    GPIO.output(RELAY_PIN, GPIO.LOW)


                else:
                    print("Small fire detected")
                # Perform action for small fire (e.g., turn on fire suppression system)
                
            #while(True):
                    if fire_center_x < frame_center_x:
                    # Fire is on the left side, so move left
                        turnright()
                        print("Move right")
                    # Perform action to move left

                    else:
                    # Fire is on the right side, so move right
                        turnleft()
                        print("Move left")
                    # Perform action to move right

                # Check if fire is above or below
                    #if fire_center_y > frame_center_y:
                    # Fire is above, so move forward
                    forward()
                    print("Move forward")
                    # Perform action to move forward

                    #else:
                    # Fire is below, so move backward
                    #    backward()
                    #    print("Move backward")
                    # Perform action to move backward

                

        fire_detected=False
        cv2.imshow('FRAME', im)
    else:
        print("Cannot read camera.")
    if cv2.waitKey(10) & 0xff == ord('q'):
        GPIO.output(RELAY_PIN, GPIO.LOW)
        break
    
GPIO.output(RELAY_PIN, GPIO.LOW)
cam.release()
cv2.destroyAllWindows()