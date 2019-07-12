import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)

cs   = 27    # board 13 pin
clk  = 18    # board 12 pin
mosi = 23    # board 16 pin
miso = 17    # board 11 pin

GPIO.setmode(GPIO.BCM)
GPIO.setup(cs,   GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(clk,  GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(mosi, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(miso, GPIO.IN)


def send_bit_0():
    GPIO.output(mosi, 0)
    GPIO.output(clk, 1)
    read = GPIO.input(miso)
    GPIO.output(clk, 0)
    return read

def send_bit_1():
    GPIO.output(mosi, 1)
    GPIO.output(clk, 1)
    read = GPIO.input(miso)
    GPIO.output(clk, 0)
    return read

def xfer(inp):
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)
    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()
    GPIO.output(cs, 1)

def xfer2(inp, inp2):
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)

    read1 = 0
    read2 = 0
    for n in range(8):
        if inp & (0x80>>n):
            read_b = send_bit_1()
        else:
            read_b = send_bit_0()
        read1 += (read_b<<(7-n))

    for n in range(8):
        if inp2 & (0x80>>n):
            read_b = send_bit_1()
        else:
            read_b = send_bit_0()
        read2 += (read_b<<(7-n))

    GPIO.output(cs, 1)
    return read1, read2

def setParam(reg, value):
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)
    for n in range(8):
        if reg & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()
    inp = 0
    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()
    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()

    for n in range(8):
        if val & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()
    GPIO.output(cs, 1)

def hard_hi_z():
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)

    # blue motor
    inp = 0b10101000
    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()

    # red motor
    inp = 0b10101000
    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()
    GPIO.output(cs, 1)

def hardstop():
    xfer2(0b10111000, 0b10111000)

def reset_device():
    xfer2(0b11000000, 0b11000000)

def get_status():
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)

    cmd = [0b11010000, 0b11010000]
    for item in cmd:
        for n in range(8):
            if item & (0x80>>n):
                send_bit_1()
            else:
                send_bit_0()

    GPIO.output(cs, 1)

def get_param(param):
    GPIO.output(clk, 0)
    GPIO.output(cs, 0)
    inp = 0b00100000 + param

    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()

    for n in range(8):
        if inp & (0x80>>n):
            send_bit_1()
        else:
            send_bit_0()

    GPIO.output(cs, 1)


def forward(inp):
    a = inp
    if a >=0:
        sign_b_0 = 1
        sign_b_1 = 0
    else:
        sign_b_0 = 0
        sign_b_1 = 1

    if a<0:
        a = -a

    xfer2(0b01000000+sign_b_0, 0b01000000+sign_b_1)   # move step
    GPIO.output(cs, 1)
    xfer2(int(a/256), int(a/256))
    xfer2(a%256, a%256)
    xfer2(0, 0)
    wait_finish()

def turn(inp):
    inp = int(inp * 2.5)
    a = inp
    if a >=0:
        sign_b_0 = 0
        sign_b_1 = 0
    else:
        sign_b_0 = 1
        sign_b_1 = 1

    if a<0:
        a = -a

    xfer2(0b01000000+sign_b_0, 0b01000000+sign_b_1)   # move step
    GPIO.output(cs, 1)
    xfer2(int(a/256), int(a/256))
    xfer2(a%256, a%256)
    xfer2(0, 0)
    wait_finish()

def wait_finish():
    for n in range(1000):
        get_param(0x19)
        xfer2(0,0)
        a,b = xfer2(0,0)
        a = a & 0x02
        b = b & 0x02
        if (a==2) & (b==2):
            break
        else:
            pass
        time.sleep(0.02)

#forward(100)
#wait_finish()
#
#time.sleep(1)
#
#turn(200)
#wait_finish()
#turn(-200)
#
#time.sleep(1)
#forward(-100)


#reset_device()
#time.sleep(0.1)

#get_status()
#time.sleep(0.1)

#hardstop()   # it works
#time.sleep(0.5)

#xfer4(0b01100000, 0, 0, 100)   # go to pos
#xfer(0b01110000)  # go home

#xfer4(0b01010000,0,0,0x41)           # RUN speed

#hard_hi_z()   # turn off bridge




#xfer2(0x80 + 0x10 + 0x02 + 1,  0x80 + 0x10 + 0x02 + 1)  #very slow move(sure work)




#MOSI--->L6470_1_SDI(RED)   L6470_1_SDO--->L6470_0_SDI   L6470_0_SD0(BLUE)--->MISO



