from scipy.interpolate import interp1d
import json
import os
import numpy as np
import pickle


def _interpolate(x_array, y_array):

    true_num = x_array.shape[0]
    feature_num = x_array.shape[1]
    interpolate_num = 80
    old_index = np.linspace(0, true_num - 1, num=true_num, endpoint=True)
    new_index = np.linspace(0, true_num - 1, num=interpolate_num, endpoint=True)
    x_new_array = np.zeros((interpolate_num, feature_num))
    y_new_array = np.zeros((interpolate_num, feature_num))
    for i in range(feature_num):
        x_list = x_array[:, i]
        y_list = y_array[:, i]
        # 扩充x
        f1 = interp1d(old_index, x_list, kind='cubic')
        new_x = f1(new_index)
        # 扩充y
        f2 = interp1d(old_index, y_list, kind='cubic')
        new_y = f2(new_index)
        x_new_array[:, i] = new_x
        y_new_array[:, i] = new_y
    return x_new_array, y_new_array


def video_interpolate():
    result_array_list = []
    result_dict = {}
    total_squats_num = []
    class_label_list = []
    video_details_list = os.path.join(data_path, 'video_details')
    for file in os.listdir(video_details_list):
        d_path = os.path.join(video_details_list, file)
        with open(d_path, encoding='utf-8') as f:
            temp_data = f.read()
            data_json = json.loads(temp_data)
        f.close()
        squats_data = data_json['squats'][0]
        in_and_out = squats_data['in_and_out']
        class_label = squats_data['class_label']
        label_num = len(class_label)
        total_squats_num.append(label_num)
        each_squats_feature_result = []

        file_name = file.split('.')[0]
        for i in range(len(class_label)):
            each_in_and_out = in_and_out[2 * i: 2 * (i + 1)]
            in_start = each_in_and_out[0]
            out_end = each_in_and_out[1]
            x_array, y_array = get_data(file_name, in_start, out_end)
            x_new_array, y_new_array = _interpolate(x_array, y_array)
            each_squats_feature_label = {}
            each_squats_feature_label['x'] = x_new_array
            each_squats_feature_label['y'] = y_new_array

            result_array_list.append(np.concatenate((x_new_array.flatten().reshape(-1, 1), y_new_array.flatten().reshape(-1,1))))

            each_squats_feature_label['class_label'] = class_label[i]
            each_squats_feature_result.append(each_squats_feature_label)

            class_label_list.append(class_label[i])
        result_dict[file_name] = each_squats_feature_result
    result_array = np.concatenate(result_array_list, axis=1)
    print(class_label_list[:5],np.array(class_label_list).shape)
    unique_labels = set(class_label_list)
    print("Unique string labels:", unique_labels)
    print("Unique string labels:", len(unique_labels))

    mapping_dir = {}
    i=0
    for name in unique_labels:
        mapping_dir[name]=i
        i+=1
    print(mapping_dir)

    mapping_dir = {name: i for i, name in enumerate(unique_labels)}
    print(mapping_dir)

    for i in range(len(class_label_list)):
        class_label_list[i] = mapping_dir[class_label_list[i]]
    print(class_label_list)

    #class_label_list = [mapping_dir[label] for label in class_label_list]
    fw = open('labels.p', 'wb')
    pickle.dump(class_label_list, fw)
    fw.close()

    binary_labels = [1 if label == 'correct' else 0 for label in class_label_list]
    print(binary_labels)
    fw = open('binary_labels.p', 'wb')
    pickle.dump(binary_labels, fw)
    fw.close()

    print('Matrix has the shape', result_array.shape)
    fw = open('squats_interpolated.p', 'wb')
    pickle.dump(result_array, fw)
    fw.close()


def get_data(file, in_start, out_end):
    num = 14
    open_pose_path = os.path.join(data_path, 'openposedata')
    file_path = os.path.join(open_pose_path, file)
    file_list = os.listdir(file_path)
    x_array = np.zeros((out_end - in_start, 18))
    y_array = np.zeros((out_end - in_start, 18))
    for file_keypoint in range(in_start, out_end):
        j = file_keypoint - in_start
        feature_origin_file = file_list[file_keypoint]
        feature_origin_path = os.path.join(file_path, feature_origin_file)
        with open(feature_origin_path, encoding='utf-8') as f:
            temp_data = f.read()
            data_json = json.loads(temp_data)
        f.close()
        pose_keypoints = data_json['people'][0]['pose_keypoints']
        right_ankle_start = (num - 1) * 3
        right_ankle_end = num * 3
        right_ankle_list = pose_keypoints[right_ankle_start: right_ankle_end]
        count = 0
        for i in range(0, len(pose_keypoints), 3):
            x_array[j][count] = pose_keypoints[i] - right_ankle_list[0]
            y_array[j][count] = pose_keypoints[i + 1] - right_ankle_list[1]
            count += 1
    return x_array, y_array


if __name__ == '__main__':
    data_path = 'C:/VScode/RoyChenProject2024/RoyChenProject2024/squats/jk_data_only_22'
    video_interpolate()
