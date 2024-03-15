## Importing third-party libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

## Read joint data
def read_skeleton(file):
    with open(file, 'r') as f: # Open the file (.skeleton)
        skeleton_sequence = {} # Initialize skeleton_sequence
        skeleton_sequence['numFrame'] = int(f.readline()) # Read the first line of the .skeleton file, which is the number of frames
        skeleton_sequence['frameInfo'] = []
        
        for t in range(skeleton_sequence['numFrame']): # Loop through each frame
            frame_info = {} # Initialize frame_info
            frame_info['numBody'] = int(f.readline()) # Call .readline function again to read the next line of the .skeleton file, which is the number of bodies
            frame_info['bodyInfo'] = []
            
            for m in range(frame_info['numBody']): # Loop through each body
                body_info = {} # Initialize body_info
                body_info_key = [ # key: The meaning represented by the number, i.e., the corresponding key
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v) # Dictionary type; key: value (float type)
                    for k, v in zip(body_info_key, f.readline().split()) # Read the next line of data, pack the data according to the key, and loop to return key, value
                }
                
                body_info['numJoint'] = int(f.readline()) # Read the next line of data, which is the number of joints
                body_info['jointInfo'] = []
                
                for v in range(body_info['numJoint']): # Loop through the data of 25 joints
                    joint_info_key = [ # Key: The meaning represented by the number, i.e., the corresponding key
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v) # Dictionary type; key: value (float type)
                        for k, v in zip(joint_info_key, f.readline().split()) # Read the next line of data, pack the data according to the key, and loop to return key, value
                    }
                    body_info['jointInfo'].append(joint_info) # Save joint data
                
                frame_info['bodyInfo'].append(body_info) # Save body data
            skeleton_sequence['frameInfo'].append(frame_info) # Save data of the current frame
    return skeleton_sequence
 
 
## Read the x, y, z coordinates of the joint, with the 0th joint position set to (0,0,0), and the other joints move together
def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file) # Call the read_skeleton() function to read the data of the .skeleton file
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body)) # Initialize data; 3 × frames × 25 × max_body
    for n, f in enumerate(seq_info['frameInfo']): # Loop through the data of each frame
        for m, b in enumerate(f['bodyInfo']): # Loop through the data of each body
            for j, v in enumerate(b['jointInfo']): # Loop through the data of each joint
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
                    data[:, n, j, m] = [v['x'], v['y'], v['z']] # Save the data of x, y, z coordinates
                else:
                    pass
    return data

def interpolate(data_path): 
    data = read_xyz(data_path)   # Read and modify the data of x, y, z coordinates
    print("shape of data:", data.shape)
    print('Read Data Done!') # Data reading is done
    if data.shape[1] < 300:
        # Calculate the number of zeros to append
        num_zeros = 300 - data.shape[1]
        # Create a zero array with the appropriate shape
        zero_array = np.zeros((data.shape[0], num_zeros, data.shape[2], data.shape[3]))
        # Append the zero array to the original data
        data = np.concatenate((data, zero_array), axis=1)
    # Check the new shape of the data
    print("New shape of data:", data.shape)
    filename = os.path.splitext(os.path.basename(data_path))[0]
    np.save(f"{filename}.npy", data)

     
def main():
    sys.path.extend(['../'])
    data_path = '../dataset/S001C001P001R001A001.skeleton' 
    interpolate(data_path) 
main()
