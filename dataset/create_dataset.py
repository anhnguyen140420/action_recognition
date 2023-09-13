import numpy as np
import pickle
import os

def generate_pose_label(list):
    data = {}
    data['pose'] = []
    data['label'] = []
    for action_ID in range(1,5):
        for subject_ID in list:
            for event_ID in range(1,6):
                file_read = open(f'keypoint/a{action_ID:0>2}_s{subject_ID:0>2}_e{event_ID:0>2}_skeleton3D.txt','r')
                pose = []
                for line in (file_read):
                    coord = line.strip().split()
                    coord.pop()
                    coord = [float(i) for i in coord]
                    pose.append(coord)
                pose = [pose[i * 33:(i + 1) * 33] for i in range((len(pose) + 33 - 1) // 33 )]
                pose = np.around(pose, 3)
                data['pose'].append(pose)
                data['label'].append(action_ID)
    return data

if __name__ == '__main__':
    train_list = [1, 2, 3, 4, 6, 7, 8, 9]
    test_list = [5]
    train = generate_pose_label(train_list)
    pickle.dump(train, open('train.pkl','wb'))
    test = generate_pose_label(test_list)
    pickle.dump(test, open('test.pkl','wb'))
