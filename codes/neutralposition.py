# Import libraries
import RPi.GPIO as GPIO
import time

# Set GPIO numbering mode
GPIO.setmode(GPIO.BOARD)

# Set pin 11 as an output, and set servo1 as pin 11 as PWM
GPIO.setup(11,GPIO.OUT)
servo1 = GPIO.PWM(11,50) # Note 11 is pin, 50 = 50Hz pulse

#start PWM running, but with value of 0 (pulse off)
servo1.start(0)
print ("Waiting for 2 seconds")
time.sleep(2)

#Let's move the servo!
print ("Rotating 180 degrees in 10 steps")

def main():
    while True:
        duty = float(input('Enter duty: '))
        if duty > 0:
            servo1.ChangeDutyCycle(duty)
        else:
            servo1.stop()
            GPIO.cleanup()
            print ("Goodbye")

if __name__ == "__main__":
    main()

