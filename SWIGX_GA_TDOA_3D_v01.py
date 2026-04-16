#Requirements:
#python 3.6.5 or 3.7.16 (also tested in 3.11.5 in VSCode via Anaconda)
#conda install clifford -c conda-forge
#pip install pyjoystick
#conda install matplotlib
#pip install mpl-toolkits.clifford

import numpy as np
#import numba as _numba

from clifford.cga import CGA, Round, Translation
from clifford.g3c import *
from clifford.tools.g3 import *
from clifford.tools.g3 import generate_rotation_rotor
from clifford.tools.g3c import *
from clifford.tools.g3c import (apply_rotor, circle_to_sphere,
                                generate_dilation_rotor,
                                generate_translation_rotor)
from clifford.tools.g3c.object_fitting import fit_circle, fit_plane
from matplotlib import pyplot as plt
from pyjoystick.sdl2 import Joystick, Key, run_event_loop
from statistics import mean 

#plt.ioff()  # we'll ask for plotting when we want it
plt.ion()
plt.show()

#import math
#import time

from mpl_toolkits.clifford import plot

import mpl_toolkits.clifford; mpl_toolkits.clifford.__version__

#import smbus


def R_euler(phi, theta,psi):
    Rphi = np.e**(-phi/2.*e12)
    Rtheta = np.e**(-theta/2.*e23)
    Rpsi = np.e**(-psi/2.*e12)

    return Rphi*Rtheta*Rpsi

def setup():
    global cga
    cga = CGA(3)

 
    #RN = 0.4*e2 - 0.4*e1
    #RS = -0.4*e2 - 0.44*e1
    #RE = 0.4*e2 + 0.44*e1
    #RW = -0.4*e2 + 0.4*e1

    #S1 = cga.round().from_center_radius(RN,1)

#    circle = cga.round(-e2, e3, e2) #unit circle at origin?

    circle = up(e1) ^ up(-e1 + e2 + e3) ^ up(-e1 - e2 - e3)


    #translator to servo4 position
    euc_translation = 1*e1
    rotor_translation = generate_translation_rotor(euc_translation)

    #scaler for servo arm lengths
    scale = generate_dilation_rotor(0.01)
    circle = (scale*circle*~scale).normal()

    euc_translation = 1*e1
    rotor_translation = generate_translation_rotor(euc_translation)
    RN = (rotor_translation*circle.normal()*~rotor_translation).normal()

    euc_translation = -1*e1
    rotor_translation = generate_translation_rotor(euc_translation)
    RS = (rotor_translation*circle.normal()*~rotor_translation).normal()

    euc_translation = 1*e2
    rotor_translation = generate_translation_rotor(euc_translation)
    RE = (rotor_translation*circle.normal()*~rotor_translation).normal()
    
    euc_translation = -1*e1-1*e2
    rotor_translation = generate_translation_rotor(euc_translation)
    RW = (rotor_translation*circle.normal()*~rotor_translation).normal()
    
    
    global fig
    global ax
    fig,ax=plt.subplots(1,1,subplot_kw=dict(projection='3d'), figsize=(8, 4))
    ax.set(xlim=[-70, 70], ylim=[-70, 70], zlim=[-70, 70])
  
    global roll
    global pitch

    roll = 0;
    pitch = 0;

    global x
    global y
    global z
    x = 30
    y = 50
    z = 1

    global A
    global B
    global C
    global D
    global A2
    global B2
    global C2
    global D2

    #8 mic locations
    A = up(e1+e2)
    B = up(-e1+e2)
    C = up(-e1-e2)
    D = up(e1-e2)

    A2 = up(e1+e2+2*e3)
    B2 = up(-e1+e2+2*e3)
    C2 = up(-e1-e2+2*e3)
    D2 = up(e1-e2+2*e3)

    ############
    ############# 
    #point distance
    #print(np.sqrt(np.abs(-2*servo11.lc(p1))))
    #print(np.sqrt(np.abs(-2*p2.lc(p1))))
    SolveTDOA()
    Render()

def Render():
    
    global fig
    global ax
    global p2show
    global p2show2
    global rendertest
    
    global S12
    global S13
    global S14
    global S12_2_1
    global S12_2
    global S13_2
    global S14_

    #ax.clear()
    #ax.set(xlim=[-70, 70], ylim=[-70, 70], zlim=[-70, 70])

    #8 mic locations in blue x
    mpl_toolkits.clifford.plot(ax, [A, B, C, D], marker='x', color='tab:blue', linestyle='none')
    mpl_toolkits.clifford.plot(ax, [A2, B2, C2, D2], marker='x', color='tab:blue', linestyle='none')
    #target
    mpl_toolkits.clifford.plot(ax, [T],  marker='X', color='tab:green', linestyle='none')

    if rendertest:
        mpl_toolkits.clifford.plot(ax, [p2show], marker='x', color='tab:red')
        mpl_toolkits.clifford.plot(ax, [p2show2], marker='x', color='tab:red')
    #SHOW CIRCLE INTERSECTIONS OF SPHERES
    #mpl_toolkits.clifford.plot(ax, [S13, S12, S14], color='tab:cyan')
    #mpl_toolkits.clifford.plot(ax, [S12_2, S13_2, S14_2], color='tab:cyan')
      
    fig
    #plt.show()
    plt.draw()
    plt.pause(0.001)

def SolveTDOA():
    
    global roll
    global pitch
    global heave
    global x
    global y
    global z
    global p2show
    global p2show2
    global rendertest
    global T
    
    step = 3
    if roll > 0.2:
        x = x + step

    if roll < -0.2:
        x = x - step

    if pitch > 0.2:
        y = y - step

    if pitch < -0.2:
        y = y + step

    if heave > 0.2:
        z = z - step

    if heave < -0.2:
        z = z + step
        
    #target
    T = up(x*e1+y*e2+z*e3)
        
    dTA = np.sqrt(np.abs(-2*T.lc(A)))
    dTB = np.sqrt(np.abs(-2*T.lc(B)))
    dTC = np.sqrt(np.abs(-2*T.lc(C)))
    dTD = np.sqrt(np.abs(-2*T.lc(D)))

    dTA2 = np.sqrt(np.abs(-2*T.lc(A2)))
    dTB2 = np.sqrt(np.abs(-2*T.lc(B2)))
    dTC2 = np.sqrt(np.abs(-2*T.lc(C2)))
    dTD2 = np.sqrt(np.abs(-2*T.lc(D2)))

    global S12
    global S13
    global S14
    global S12_2_1
    global S12_2
    global S13_2
    global S14_2
    
   
    for scaleR in range(40,120,1):
    
        S1 = cga.round().from_center_radius(A,dTA*scaleR/100)
        S2 = cga.round().from_center_radius(B,dTB*scaleR/100)
        S3 = cga.round().from_center_radius(C,dTC*scaleR/100)
        S4 = cga.round().from_center_radius(D,dTD*scaleR/100)
 
        S1_2 = cga.round().from_center_radius(A2,dTA2*scaleR/100)
        S2_2 = cga.round().from_center_radius(B2,dTB2*scaleR/100)
        S3_2 = cga.round().from_center_radius(C2,dTC2*scaleR/100)
        S4_2 = cga.round().from_center_radius(D2,dTD2*scaleR/100)

        #CIRCLE INTERSECTIONS OF SPHERES
        #S12 = meet(S1.mv, S2.mv)
        #S13 = meet(S1.mv, S3.mv)
        #S14 = meet(S1.mv, S4.mv)
        #S12_2_1 = meet(S1.mv, S1_2.mv)
        #S12_2 = meet(S1.mv, S2_2.mv)
        #S13_2 = meet(S1.mv, S3_2.mv)
        #S14_2 = meet(S1.mv, S4_2.mv)
        
        #intersect two pair of three spheres, and get resulting point pairs              
        pt1,pt2 = point_pair_to_end_points(fast_dual(fast_dual(S1.mv) ^ fast_dual(S2.mv) ^ fast_dual(S3_2.mv)))
        pt3,pt4 = point_pair_to_end_points(fast_dual(fast_dual(S1_2.mv) ^ fast_dual(S2_2.mv) ^ fast_dual(S3.mv)))

        #check distances between points
        dT12 = np.abs(np.sqrt(np.abs(-2*pt1.lc(pt2))))
        dT13 = np.abs(np.sqrt(np.abs(-2*pt1.lc(pt3))))
        dT14 = np.abs(np.sqrt(np.abs(-2*pt1.lc(pt4))))
        dT23 = np.abs(np.sqrt(np.abs(-2*pt2.lc(pt3))))
        dT24 = np.abs(np.sqrt(np.abs(-2*pt2.lc(pt4))))
        dT34 = np.abs(np.sqrt(np.abs(-2*pt3.lc(pt4))))

        rendertest = False
        dthres = 0.25
        name = 'clear'
        if dT12 < dthres:
            p2show = ((pt1(1)))
            p2show2 = ((pt2(1)))
            rendertest = True
            name = 'dT12'
            break
  
        if dT13 < dthres:
            p2show = ((pt1(1)))
            p2show2 = ((pt3(1)))
            rendertest = True
            name = 'dT13'
            break
 
        if dT14 < dthres:
            p2show = ((pt1(1)))
            p2show2 = ((pt4(1)))
            rendertest = True
            name = 'dT14'
            break

        if dT23 < dthres:
            p2show = ((pt2(1)))
            p2show2 = ((pt3(1)))
            rendertest = True
            name = 'dT23'
            break

        if dT24 < dthres:
            p2show = ((pt2(1)))
            p2show2 = ((pt4(1)))
            rendertest = True
            name = 'dT24'
            break
 
        if dT34 < dthres:
            p2show = ((pt3(1)))
            p2show2 = ((pt4(1)))
            rendertest = True
            name = 'dT34'
            break

        #if rendertest:
        #    print(np.abs(np.sqrt(np.abs(-2*p2show.lc(p2show2)))))
        #    print(name)
            

    Render()


roll = 0
pitch = 0
yaw = 0
heave = 1

count = 0
oldroll = 0
oldpitch = 0
oldyaw = 0
oldheave = 0

setup()

if True:
    import argparse
    import time

    P = argparse.ArgumentParser(description='Run a thread event loop.')
    P.add_argument('--lib', type=str, default='sdl', choices=['sdl', 'pygame'],
                   help='Library to run the thread event loop with.')
    P.add_argument('--timeout', type=float, default=float('inf'), help='Time to run for')
    P.add_argument('--keytype', type=str, default=None, choices=[Key.AXIS, ])

    devices = Joystick.get_joysticks()
    print("Devices:", devices)

    if len(devices) == 0:
        monitor = None
        print('No joystick detected. Running without joystick monitor input.')
    else:
        monitor = devices[0]
    monitor_keytypes = [Key.AXIS]
     
    def print_add(joy):
        print('Added', joy, '\n', end='\n', flush=True)

    def print_remove(joy):
        print('Removed', joy, '\n', end='\n', flush=True)

    def key_received(key):
        global roll,pitch,yaw,heave
        global oldroll,oldpitch,oldyaw,oldheave
        
        if monitor is None:
            print(key, '==', key.value)
        elif key.joystick == monitor:
            monitor.update_key(key)
            for k in monitor.keys:
                if k.keytype in monitor_keytypes:
                    format_key(k)
                    
                    if abs(roll-oldroll)>0.1 or abs(pitch-oldpitch)>0.1 or abs(yaw-oldyaw)>0.1 or abs(heave-oldheave)>0.1:
                        #print(roll)
                        oldroll = roll
                        oldpitch = pitch
                        oldyaw = yaw
                        oldheave = heave
                        SolveTDOA()
                        #Render()
            

    def format_key(key):
                    
        global roll
        global pitch
        global yaw
        global heave
        
              
        if key.number == 0:
            roll = key.value
            #print(roll)
        if key.number == 1:
            pitch = key.value
        if key.number == 2:
            heave = key.value
            #print(heave)
        if key.number == 3:
            yaw = key.value 

    if monitor is None:
        print('Keyboard control enabled (click plot window first): Arrow keys')

        def on_key_press(event):
            global x, y

            step = 3
            if event.key is None:
                return

            key = event.key.lower()
            if key == 'up':
                y = y + step
            elif key == 'down':
                y = y - step
            elif key == 'left':
                x = x - step
            elif key == 'right':
                x = x + step
            else:
                return

            SolveTDOA()

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Keep the figure alive so matplotlib keyboard events continue to be handled.
        try:
            while plt.fignum_exists(fig.number):
                plt.pause(0.05)
        except KeyboardInterrupt:
            pass
    else:
        run_event_loop(print_add, print_remove, key_received)

