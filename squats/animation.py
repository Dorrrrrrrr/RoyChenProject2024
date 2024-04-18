import os  
import json  
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation  
  

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

fig, ax = plt.subplots()  
ax.set_xlim(-1000, 1080)  
ax.set_ylim(0, 1080)  
ax.invert_yaxis()  

lines = []  
scatters = []  

def load_data(file_path):   
    if not os.path.exists(file_path):  
        return None  
    with open(file_path, 'r') as file:  
        data = json.load(file)  
    pose_keypoints = data['people'][0]['pose_keypoints']  
    x, y, c = pose_keypoints[::3], pose_keypoints[1::3], pose_keypoints[2::3]  

    offset = x[13]
    for i in range(len(x)):
        if x[i] != 0 and y[i] != 0:
            x[i] = x[i] - offset

    return x, y  


def init():  
    for line in lines:  
        line.set_data([], [])  
    for scatter in scatters:  
        scatter.set_offsets([(0, 0)])  
    return lines + scatters  

def update(frame):  

    lines = []
    scatters = []


    x, y = load_data(frame)  
    if x is None:  
        return lines + scatters  

    for i, (xi, yi) in enumerate(zip(x, y)):  
        if xi == 0 and yi == 0:  
            continue  
        if i < len(scatters):  
            scatters[i].set_offsets([(xi, yi)])  
        else:  
            scatters.append(ax.scatter([xi], [yi], color='blue'))  
  
    for i, conn in enumerate(connections):  
        start_point = (x[conn[0]], y[conn[0]])  
        end_point = (x[conn[1]], y[conn[1]])  
        if (start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):  
            continue

        line, = ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')
        lines.append(line) 

    return lines + scatters  
  

foler_path = 'C:/VScode/RoyChenProject2024/RoyChenProject2024/squats/jk_data_only_22//openposedata/J_01'
frames = sorted([os.path.join(foler_path, file) for file in os.listdir(foler_path)])
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)  
  
plt.show()