from gpiozero import Motor, DistanceSensor, AngularServo
from time import sleep

sensor = DistanceSensor(24, 18)
servo = AngularServo(17, min_angle=-90, max_angle=90)
motor = Motor(forward=7, backward=8)

## change once full chassis is installed
## functions defined for testing 

def fix_forward():
    motor.forward()
    sleep(1)
    motor.stop()

def fix_backward():
    motor.forward()
    motor.reverse()
    sleep(1)
    motor.stop()

def fix_left():
    ## once chassis installed, actually turn left using Robot class
    motor.forward()
    motor.reverse()
    sleep(1)
    motor.stop()

def fix_right():
    ## once chassis installed, actually turn right using Robot class
    motor.forward()
    motor.reverse()
    sleep(1)
    motor.stop()

def close_arm():
    servo.angle = 90
    sleep(1)

def open_arm():
    servo_angle = -90
    sleep(1)
    
