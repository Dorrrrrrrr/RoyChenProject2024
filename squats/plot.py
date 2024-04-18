import os
import json
import matplotlib.pyplot as plt

connections = [  
    [0, 1],       # Nose to Neck
    [1, 2],       # Neck to Right Shoulder
    [2, 3],       # Right Shoulder to Right Elbow
    [3, 4],       # Right Elbow to Right Wrist
    [1, 5],       # Neck to Left Shoulder
    [5, 6],       # Left Shoulder to Left Elbow
    [6, 7],       # Left Elbow to Left Wrist
    [1, 8],       # Neck to Right Hip
    [8, 9],       # Right Hip to Right Knee
    [9, 10],      # Right Knee to Right Ankle
    [1, 11],      # Neck to Left Hip
    [11, 12],     # Left Hip to Left Knee
    [12, 13],     # Left Knee to Left Ankle
    [0, 14],      # Nose to Right Eye
    [0, 15],      # Nose to Left Eye
    [14, 16],     # Nose to Right Ear
    [15, 17]      # Nose to Left Ear
]

file_path = 'C:/VScode/RoyChenProject2024/RoyChenProject2024/squats/jk_data_only_22/openposedata/a/DSC_0808_000000000000_keypoints.json'

with open(file_path, 'r') as file:  
    data = json.load(file)  

pose_keypoints = data['people'][0]['pose_keypoints']
x, y, c = pose_keypoints[::3], pose_keypoints[1::3], pose_keypoints[2::3]


right_ankle_x, right_ankle_y = x[10], y[10]


x = [x_i - right_ankle_x for x_i in x]
y = [y_i - right_ankle_y for y_i in y]

for conn in connections:
    
    if c[conn[0]] != 0 and c[conn[1]] != 0:
        start_point = (x[conn[0]], y[conn[0]])
        end_point = (x[conn[1]], y[conn[1]])
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')


for i, (x_i, y_i, c_i) in enumerate(zip(x, y, c)):
    
    if c_i != 0:
        plt.scatter(x_i, y_i, color='blue')
        plt.annotate(str(i), (x_i, y_i), textcoords="offset points", xytext=(0,10), ha='center')
    

for i, (x_i, y_i, c_i) in enumerate(zip(x, y, c)):
    
    if c_i != 0:
        plt.scatter(x_i, y_i, color='blue')
        plt.annotate(str(i), (x_i, y_i), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlim(-600, 600)
plt.ylim(-600, 600)
plt.gca().invert_yaxis()
plt.show() 