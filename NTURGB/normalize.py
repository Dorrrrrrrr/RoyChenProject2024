## Import third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
 
 
## Read joint data
def read_skeleton(file):
    with open(file, 'r') as f: # Open the file(.skeleton)
        skeleton_sequence = {} # Initialize skeleton_sequence
        skeleton_sequence['numFrame'] = int(f.readline()) # Read the first line of the .skeleton file, which is the number of frames
        skeleton_sequence['frameInfo'] = []
        
        for t in range(skeleton_sequence['numFrame']): # Iterate through each frame
            frame_info = {} # Initialize frame_info
            frame_info['numBody'] = int(f.readline()) # Read the next line of the .skeleton file, which is the number of bodies
            frame_info['bodyInfo'] = []
            
            for m in range(frame_info['numBody']): # Iterate through each body
                body_info = {} # Initialize body_info
                body_info_key = [ # key: numerical representation meaning the corresponding key
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v) # Dictionary type; key: value(float type)
                    for k, v in zip(body_info_key, f.readline().split()) # Read the next line of data, pack data according to key, iterate to return key, value
                }
                
                body_info['numJoint'] = int(f.readline()) # Read the next line of data, which is the number of joints
                body_info['jointInfo'] = []
                
                for v in range(body_info['numJoint']): # Iterate through the data of 25 joints
                    joint_info_key = [ # Key: numerical representation meaning the corresponding key
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v) # Dictionary type; key: value(float type)
                        for k, v in zip(joint_info_key, f.readline().split()) # Read the next line of data, pack data according to key, iterate to return key, value
                    }
                    body_info['jointInfo'].append(joint_info) # Save joint data
                
                frame_info['bodyInfo'].append(body_info) # Save body data
            skeleton_sequence['frameInfo'].append(frame_info) # Save data for the current frame
    return skeleton_sequence
 
 
## Read the x, y, z coordinates of joints, place joint 0 at (0,0,0), and coordinate transformation for other joints
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file) # Call the read_skeleton() function to read the data from the .skeleton file
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body)) # Initialize data; 3 × number of frames × 25 × max_body  
    for n, f in enumerate(seq_info['frameInfo']): # Iterate through the data for each frame
        for m, b in enumerate(f['bodyInfo']): # Iterate through the data for each body
            for j, v in enumerate(b['jointInfo']): # Iterate through the data for each joint
                if j == 0:
                    diff_x = v['x']
                    diff_y = v['y']
                    diff_z = v['z']
                    v['x'] = 0
                    v['y'] = 0
                    v['z'] = 0
                else:
                    v['x'] = v['x'] - diff_x
                    v['y'] = v['y'] - diff_y
                    v['z'] = v['z'] - diff_z
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']] # Save data for x, y, z coordinates
                else:
                    pass
    return data
 
 
## 2D visualization
def Print2D(num_frame, point, arms, rightHand, leftHand, legs, body):
    
    # Find the maximum coordinates
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :]) 
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])
    
    n = 0     # Start displaying from frame n
    m = num_frame   # End at frame m, n < m < row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla() # Clear axis, i.e., clear the current active axis in the current figure, other axes are not affected
 
        # Draw all joints of two bodies
        #plt.scatter(point[0, i, :, :], point[1, i, :, :], c='red', s=40.0) # c: color;  s: size
        # One body
        plt.scatter(point[0, i, :, 0], point[1, i, :, 0], c='red', s=40.0)
        # Connect joints of the first body to form skeleton
        plt.plot(point[0, i, arms, 0], point[1, i, arms, 0], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c='green', lw=2.0) # c: color;  lw: line width 
        plt.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 0], point[1, i, legs, 0], c='green', lw=2.0)
        plt.plot(point[0, i, body, 0], point[1, i, body, 0], c='green', lw=2.0)
 
                # Connect the joints of the second body to form the skeleton
        plt.plot(point[0, i, arms, 1], point[1, i, arms, 1], c='green', lw=2.0)
        plt.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c='green', lw=2.0)
        plt.plot(point[0, i, legs, 1], point[1, i, legs, 1], c='green', lw=2.0)
        plt.plot(point[0, i, body, 1], point[1, i, body, 1], c='green', lw=2.0)
 
        plt.text(xmax, ymax+0.2, 'frame: {}/{}'.format(i, num_frame-1)) # Text explanation
        plt.xlim(xmin-0.5, xmax+0.5) # x-axis range
        plt.ylim(ymin-0.3, ymax+0.3) # y-axis range
        plt.pause(0.001) # Pause delay
 
    plt.ioff() 
    plt.savefig('2d_norm.png')
    plt.show()
 
 
## 3D display    
def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body):
 
    # Calculate maximum coordinates
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :]) 
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])    
    
    n = 0     # Start displaying from frame n
    m = num_frame   # End at frame m, n<m<row
    plt.figure()
    plt.ion()
    for i in range(n, m):
        plt.cla() # Clear axis, i.e., clear the current active axis in the current figure, other axes are not affected
 
        plot3D = plt.subplot(projection='3d')
        plot3D.view_init(120, -90) # Change the viewing angle
        
        Expan_Multiple = 1.4 # Coordinate expansion factor for aesthetic plotting
        
        # Plot all joints of two bodies
        #plot3D.scatter(point[0, i, :, :]*Expan_Multiple, point[1, i, :, :]*Expan_Multiple, point[2, i, :, :], c='red', s=40.0) # c: color;  s: size
        # One body
        plt.scatter(point[0, i, :, 0], point[1, i, :, 0], c='red', s=40.0)
        # Connect the joints of the first body to form the skeleton
        plot3D.plot(point[0, i, arms, 0]*Expan_Multiple, point[1, i, arms, 0]*Expan_Multiple, point[2, i, arms, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0]*Expan_Multiple, point[1, i, rightHand, 0]*Expan_Multiple, point[2, i, rightHand, 0], c='green', lw=2.0) # c: color;  lw: line width 
        plot3D.plot(point[0, i, leftHand, 0]*Expan_Multiple, point[1, i, leftHand, 0]*Expan_Multiple, point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0]*Expan_Multiple, point[1, i, legs, 0]*Expan_Multiple, point[2, i, legs, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0]*Expan_Multiple, point[1, i, body, 0]*Expan_Multiple, point[2, i, body, 0], c='green', lw=2.0)
 
        # Connect the joints of the second body to form the skeleton
        plot3D.plot(point[0, i, arms, 1]*Expan_Multiple, point[1, i, arms, 1]*Expan_Multiple, point[2, i, arms, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 1]*Expan_Multiple, point[1, i, rightHand, 1]*Expan_Multiple, point[2, i, rightHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 1]*Expan_Multiple, point[1, i, leftHand, 1]*Expan_Multiple, point[2, i, leftHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 1]*Expan_Multiple, point[1, i, legs, 1]*Expan_Multiple, point[2, i, legs, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 1]*Expan_Multiple, point[1, i, body, 1]*Expan_Multiple, point[2, i, body, 1], c='green', lw=2.0)
 
        plot3D.text(xmax-0.3, ymax+1.1, zmax+0.3, 'frame: {}/{}'.format(i, num_frame-1)) # Text explanation
        plot3D.set_xlim3d(xmin-0.5, xmax+0.5) # x-axis range
        plot3D.set_ylim3d(ymin-0.3, ymax+0.3) # y-axis range
        plot3D.set_zlim3d(zmin-0.3, zmax+0.3) # z-axis range
        plt.pause(0.001) # Pause delay
 
    plt.ioff() 
    plt.savefig('3d_norm.png')
    plt.show() 
    
 
## main function
def main():
    sys.path.extend(['../']) # Extend path
    data_path = '../dataset/S001C001P001R001A001.skeleton' 
    point = read_xyz(data_path)   # Read x, y, z coordinates
    print('Read Data Done!') # Data reading completed
 
    num_frame = point.shape[1] # Number of frames
    print(point.shape)  # Number of coordinates (3) × Number of frames × Number of joints (25) × max_body(2)
 
    # Adjacent joint indices
    arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21] # 23 <-> 11 <-> 10 ...
    rightHand = [11, 24] # 11 <-> 24
    leftHand = [7, 22] # 7 <-> 22
    legs = [19, 18, 17, 16, 0, 12, 13, 14, 15] # 19 <-> 18 <-> 17 ...
    body = [3, 2, 20, 1, 0]  # 3 <-> 2 <-> 20 ...
    
    Print2D(num_frame, point, arms, rightHand, leftHand, legs, body)  # 2D visualization
    #Print3D(num_frame, point, arms, rightHand, leftHand, legs, body)
 
main()

