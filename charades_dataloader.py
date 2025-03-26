import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import os
import os.path
from tqdm import tqdm
import random
from utils import *
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn


def make_dataset(split_file, split, root, num_classes=15):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!', split)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid + '.npy')):
            continue

        if len(data[vid]['actions']) < 1:
            continue

        fts = np.load(os.path.join(root, vid + '.npy'))
        num_feat = fts.shape[0]
        label = np.zeros((num_feat, num_classes), np.float32)
        #
        hmap = np.zeros((num_feat, num_classes), np.float32)
        action_lengths = []
        center_loc = []
        num_action = 0

        fps = num_feat / data[vid]['duration']
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr+1) / fps > mid_point and fr / fps < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration = int((ann[2] - ann[1]) * fps)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])

        dataset.append((vid, label, data[vid]['duration'], [hmap, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        i += 1

    return dataset



class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root_rgb, root_flow, root_depth, root_pose, root_SAM, root_VLM, batch_size, classes, num_clips, skip):

        self.data_rgb = make_dataset(split_file, split, root_rgb, classes)
        self.split = split
        self.split_file = split_file
        self.batch_size = batch_size
        self.root_rgb = root_rgb
        self.root_flow = root_flow
        self.root_depth = root_depth
        self.root_pose = root_pose
        self.root_SAM = root_SAM
        self.root_VLM = root_VLM
        self.in_mem = {}
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        ################3RGB#########################################
        entry_rgb = self.data_rgb[index]
        feat_rgb = np.load(os.path.join(self.root_rgb, entry_rgb[0] + '.npy'))


        labels_rgb = entry_rgb[1]

        hmap_rgb, num_action_rgb, center_loc_rgb, action_lengths_rgb = entry_rgb[3]
    ##########################RGB##################################################################3

    ####################################################flow#########################################


        feat_flow = np.load(os.path.join(self.root_flow, entry_rgb[0] + '.npy'))




    ##########################flow##################################################################3
    ####################################################depth#########################################

        feat_depth = np.load(os.path.join(self.root_depth, entry_rgb[0] + '.npy'))


        feat_depth =  np.mean(feat_depth, axis=1)





    ##########################depth##################################################################3
    ####################################################POSE#########################################

        feat_pose = np.load(os.path.join(self.root_pose, entry_rgb[0] + '.npy'))

        feat_pose =  np.mean(feat_pose, axis=1)


    ##########################POSE##################################################################3
    ####################################################SAM#########################################

        feat_SAM = np.load(os.path.join(self.root_SAM, entry_rgb[0] + '.npy'))

        feat_SAM = np.mean(feat_SAM, axis=1)

    ##########################SAM##################################################################3
    ####################################################VLM#########################################

        feat_VLM = np.load(os.path.join(self.root_VLM, entry_rgb[0] + '.npy'))

        feat_VLM = np.mean(feat_VLM, axis=1)


    ##########################VLM##################################################################3


        # Ensure num_clips does not exceed the minimum feature length
        num_clips = self.num_clips

        feat_rgb = feat_rgb.reshape((feat_rgb.shape[0], 1, 1, feat_rgb.shape[-1]))
        features_rgb = feat_rgb.astype(np.float32)
        feat_flow = feat_flow.reshape((feat_flow.shape[0], 1, 1, feat_flow.shape[-1]))
        features_flow = feat_flow.astype(np.float32)
        feat_depth = feat_depth.reshape((feat_depth.shape[0], 1, 1, feat_depth.shape[-1]))
        features_depth = feat_depth.astype(np.float32)
        feat_pose = feat_pose.reshape((feat_pose.shape[0], 1, 1, feat_pose.shape[-1]))
        features_pose = feat_pose.astype(np.float32)
        feat_SAM = feat_SAM.reshape((feat_SAM.shape[0], 1, 1, feat_SAM.shape[-1]))
        features_SAM = feat_SAM.astype(np.float32)
        feat_VLM = feat_VLM.reshape((feat_VLM.shape[0], 1, 1, feat_VLM.shape[-1]))
        features_VLM = feat_VLM.astype(np.float32)


        # Randomly select clips for training/testing
        if self.split in ["training"]:
            if len(feat_rgb) > num_clips and num_clips > 0:


                random_index = random.choice(range(0, len(features_rgb) - num_clips))

                # Slice features, labels, and heatmaps for each modality
                features_rgb = features_rgb[random_index: random_index + num_clips: 1]
                labels_rgb = labels_rgb[random_index: random_index + num_clips: 1]
                hmap_rgb = hmap_rgb[random_index: random_index + num_clips: 1]

                features_flow = features_flow[random_index: random_index + num_clips: 1]


                features_depth = features_depth[random_index: random_index + num_clips: 1]


                features_pose = features_pose[random_index: random_index + num_clips: 1]


                features_SAM = features_SAM[random_index: random_index + num_clips: 1]


                features_VLM = features_VLM[random_index: random_index + num_clips: 1]



        else:

            random_index = 0

            # Slice features, labels, and heatmaps for each modality
            features_rgb = features_rgb[random_index: random_index + num_clips: 1]
            labels_rgb = labels_rgb[random_index: random_index + num_clips: 1]
            hmap_rgb = hmap_rgb[random_index: random_index + num_clips: 1]

            features_flow = 0


            features_depth = 0


            features_pose = 0


            features_SAM = 0


            features_VLM = 0


        return features_rgb, labels_rgb, hmap_rgb, action_lengths_rgb, [entry_rgb[0], entry_rgb[2], num_action_rgb], features_flow, features_depth, features_pose, features_SAM, features_VLM

    def __len__(self):
        return len(self.data_rgb)


class collate_fn_unisize():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        new_batch = []

        for b in batch:


            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            h = np.zeros((max_len, b[2].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            h[:b[0].shape[0], :] = b[2]

            f1 = np.zeros((max_len, b[5].shape[1], b[5].shape[2], b[5].shape[3]), np.float32)

            f1[:b[5].shape[0]] = b[5]

            f2 = np.zeros((max_len, b[6].shape[1], b[6].shape[2], b[6].shape[3]), np.float32)

            f2[:b[6].shape[0]] = b[6]


            f3 = np.zeros((max_len, b[7].shape[1], b[7].shape[2], b[7].shape[3]), np.float32)

            f3[:b[7].shape[0]] = b[7]


            f4 = np.zeros((max_len, b[8].shape[1], b[8].shape[2], b[8].shape[3]), np.float32)

            f4[:b[8].shape[0]] = b[8]


            f5 = np.zeros((max_len, b[9].shape[1], b[9].shape[2], b[9].shape[3]), np.float32)

            f5[:b[9].shape[0]] = b[9]


            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), video_to_tensor(f1), video_to_tensor(f2), video_to_tensor(f3), video_to_tensor(f4), video_to_tensor(f5)])

        return default_collate(new_batch)
class collate_fn_unisize_eval():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize_eval(self, batch):
        max_len = int(self.num_clips)

        new_batch = []

        for b in batch:


            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            h = np.zeros((max_len, b[2].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            h[:b[0].shape[0], :] = b[2]

            f1 = 0


            f2 = 0



            f3 = 0



            f4 = 0


            f5 = 0


            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), f1,f2,f3, f4,f5])

        return default_collate(new_batch)

