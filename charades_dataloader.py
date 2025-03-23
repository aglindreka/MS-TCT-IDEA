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

# def pad_repeat_last(modality, max_length):
#     current_length = modality.shape[0]
#     if current_length < max_length:
#         last_frame = modality[-1:]  # Get last frame
#         repeat_frames = np.repeat(last_frame, max_length - current_length, axis=0)  # Repeat it
#         modality = np.vstack([modality, repeat_frames])
#     return modality[:max_length]  # Truncate if too long
def make_dataset(split_file, split, root_rgb, root_flow, root_depth, root_pose, root_SAM, root_VLM, num_classes=15):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset_rgb = []
    dataset_flow = []
    dataset_depth = []
    dataset_pose = []
    dataset_SAM = []
    dataset_VLM = []

    with open(split_file, 'r') as f:
        data = json.load(f)
    print('split!!!!', split)
    i = 0
    for vid in tqdm(data.keys()):
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root_rgb, vid + '.npy')):
            continue

        if len(data[vid]['actions']) < 1:
            continue

        fts_rgb = np.load(os.path.join(root_rgb, vid + '.npy'))



        fts_flow = np.load(os.path.join(root_flow, vid + '.npy'))


        fts_depth = np.load(os.path.join(root_depth, vid + '.npy'))

        fts_pose = np.load(os.path.join(root_pose, vid + '.npy'))

        fts_SAM = np.load(os.path.join(root_SAM, vid + '.npy'))

        fts_VLM = np.load(os.path.join(root_VLM, vid + '.npy'))


        num_feat_rgb = fts_rgb.shape[0]

        num_feat_flow = fts_flow.shape[0]
        num_feat_depth = fts_depth.shape[0]
        num_feat_pose = fts_pose.shape[0]
        num_feat_SAM = fts_SAM.shape[0]
        num_feat_VLM = fts_VLM.shape[0]



        label_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        label_flow = np.zeros((num_feat_flow, num_classes), np.float32)
        label_depth = np.zeros((num_feat_depth, num_classes), np.float32)
        label_pose = np.zeros((num_feat_pose, num_classes), np.float32)
        label_SAM = np.zeros((num_feat_SAM, num_classes), np.float32)
        label_VLM = np.zeros((num_feat_VLM, num_classes), np.float32)

        #
        hmap_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        hmap_flow = np.zeros((num_feat_flow, num_classes), np.float32)
        hmap_depth = np.zeros((num_feat_depth, num_classes), np.float32)
        hmap_pose = np.zeros((num_feat_pose, num_classes), np.float32)
        hmap_SAM = np.zeros((num_feat_SAM, num_classes), np.float32)
        hmap_VLM = np.zeros((num_feat_VLM, num_classes), np.float32)

        action_lengths = []
        center_loc = []

        action_lengths_flow = []
        center_loc_flow = []

        action_lengths_depth = []
        center_loc_depth = []

        action_lengths_pose = []
        center_loc_pose = []

        action_lengths_SAM = []
        center_loc_SAM = []

        action_lengths_VLM = []
        center_loc_VLM = []


        num_action = 0
        num_action_flow = 0
        num_action_depth = 0
        num_action_pose = 0
        num_action_SAM = 0
        num_action_VLM = 0

        fps_rgb = num_feat_rgb / data[vid]['duration']
        fps_flow = num_feat_flow / data[vid]['duration']
        fps_depth = num_feat_depth / data[vid]['duration']
        fps_pose = num_feat_pose / data[vid]['duration']
        fps_SAM = num_feat_SAM / data[vid]['duration']
        fps_VLM = num_feat_VLM / data[vid]['duration']

        #################RGB##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_rgb, 1):
                if fr / fps_rgb > ann[1] and fr / fps_rgb< ann[2]:
                    label_rgb[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_rgb > mid_point and fr / fps_rgb < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration = int((ann[2] - ann[1]) * fps_rgb)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap_rgb[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])
        #################RGB##################################################################3

        #################flow##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_flow, 1):
                if fr / fps_flow > ann[1] and fr / fps_flow< ann[2]:
                    label_flow[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_flow > mid_point and fr / fps_flow < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration_flow = int((ann[2] - ann[1]) * fps_flow)
                    radius = int(action_duration_flow / gamma)
                    generate_gaussian(hmap_flow[:, class_], center, radius, tau, ku)
                    num_action_flow = num_action + 1
                    center_loc_flow.append([center, class_])
                    action_lengths_flow.append([action_duration])
        #################flow##################################################################3
        #################DEPTH##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_depth, 1):
                if fr / fps_depth > ann[1] and fr / fps_depth < ann[2]:
                    label_depth[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_depth > mid_point and fr / fps_depth < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration_depth = int((ann[2] - ann[1]) * fps_depth)
                    radius = int(action_duration_depth / gamma)
                    generate_gaussian(hmap_depth[:, class_], center, radius, tau, ku)
                    num_action_depth = num_action_depth + 1
                    center_loc_depth.append([center, class_])
                    action_lengths_depth.append([action_duration_depth])
        #################Depth##################################################################3

        #################POSE##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_pose, 1):
                if fr / fps_pose > ann[1] and fr / fps_pose < ann[2]:
                    label_pose[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_pose > mid_point and fr / fps_pose < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration_pose = int((ann[2] - ann[1]) * fps_pose)
                    radius = int(action_duration_pose / gamma)
                    generate_gaussian(hmap_pose[:, class_], center, radius, tau, ku)
                    num_action_pose = num_action + 1
                    center_loc_pose.append([center, class_])
                    action_lengths_pose.append([action_duration_pose])
        #################POSE##################################################################3

        #################SAM##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_SAM, 1):
                if fr / fps_SAM > ann[1] and fr / fps_SAM < ann[2]:
                    label_SAM[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_SAM > mid_point and fr / fps_SAM < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration_SAM = int((ann[2] - ann[1]) * fps_SAM)
                    radius = int(action_duration_SAM / gamma)
                    generate_gaussian(hmap_SAM[:, class_], center, radius, tau, ku)
                    num_action_SAM = num_action_SAM + 1
                    center_loc_SAM.append([center, class_])
                    action_lengths_SAM.append([action_duration_SAM])
        #################SAM##################################################################3

        #################VLM##################################################################3
        for ann in data[vid]['actions']:
            #
            if ann[2] < ann[1]:
                continue
            mid_point = (ann[2] + ann[1]) / 2
            for fr in range(0, num_feat_VLM, 1):
                if fr / fps_VLM > ann[1] and fr / fps_VLM < ann[2]:
                    label_VLM[fr, ann[0]] = 1  # binary classification

                # G* Ground truth Heat-map
                # if fr / fps + 1 > mid_point and fr / fps < mid_point:
                if (fr + 1) / fps_VLM > mid_point and fr / fps_VLM < mid_point:
                    center = fr + 1
                    class_ = ann[0]
                    action_duration_VLM = int((ann[2] - ann[1]) * fps_VLM)
                    radius = int(action_duration_VLM / gamma)
                    generate_gaussian(hmap_VLM[:, class_], center, radius, tau, ku)
                    num_action_VLM = num_action_VLM + 1
                    center_loc_VLM.append([center, class_])
                    action_lengths_VLM.append([action_duration_VLM])
        #################VLM##################################################################3



        dataset_rgb.append(
            (vid, label_rgb, data[vid]['duration'], [hmap_rgb, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        dataset_flow.append(
            (vid, label_flow, data[vid]['duration'], [hmap_flow, num_action_flow, np.asarray(center_loc_flow), np.asarray(action_lengths_flow)]))
        dataset_depth.append(
            (vid, label_depth, data[vid]['duration'],
             [hmap_depth, num_action_depth, np.asarray(center_loc_depth), np.asarray(action_lengths_depth)]))
        dataset_pose.append(
            (vid, label_pose, data[vid]['duration'],
             [hmap_pose, num_action_pose, np.asarray(center_loc_pose), np.asarray(action_lengths_pose)]))
        dataset_SAM.append(
            (vid, label_SAM, data[vid]['duration'],
             [hmap_SAM, num_action_SAM, np.asarray(center_loc_SAM), np.asarray(action_lengths_SAM)]))
        dataset_VLM.append(
            (vid, label_VLM, data[vid]['duration'],
             [hmap_VLM, num_action_VLM, np.asarray(center_loc_VLM), np.asarray(action_lengths_VLM)]))
        i += 1


    return dataset_rgb, dataset_flow, dataset_depth, dataset_pose, dataset_SAM, dataset_VLM


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root_rgb, root_flow, root_depth, root_pose, root_SAM, root_VLM, batch_size, classes, num_clips, skip):

        self.data_rgb, self.data_flow, self.data_depth, self.data_pose, self.data_SAM, self.data_VLM = make_dataset(split_file, split, root_rgb, root_flow, root_depth, root_pose, root_SAM, root_VLM, classes)

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
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################RGB##################################################################3

    ####################################################flow#########################################
        entry_flow = self.data_flow[index]
        feat_flow = np.load(os.path.join(self.root_flow, entry_flow[0] + '.npy'))



        labels_flow = entry_flow[1]

        hmap_flow, num_action_flow, center_loc_flow, action_lengths_flow = entry_flow[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])


    ##########################flow##################################################################3
    ####################################################depth#########################################
        entry_depth = self.data_depth[index]
        feat_depth = np.load(os.path.join(self.root_depth, entry_depth[0] + '.npy'))

        feat_depth =  np.mean(feat_depth, axis=1)

        labels_depth = entry_depth[1]

        hmap_depth, num_action_depth, center_loc_depth, action_lengths_depth = entry_depth[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])



    ##########################depth##################################################################3
    ####################################################POSE#########################################
        entry_pose = self.data_pose[index]
        feat_pose = np.load(os.path.join(self.root_pose, entry_pose[0] + '.npy'))

        feat_pose =  np.mean(feat_pose, axis=1)


        labels_pose = entry_pose[1]

        hmap_pose, num_action_pose, center_loc_pose, action_lengths_pose = entry_pose[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################POSE##################################################################3
    ####################################################SAM#########################################
        entry_SAM = self.data_SAM[index]
        feat_SAM = np.load(os.path.join(self.root_SAM, entry_SAM[0] + '.npy'))

        feat_SAM = np.mean(feat_SAM, axis=1)


        labels_SAM = entry_SAM[1]

        hmap_SAM, num_action_SAM, center_loc_SAM, action_lengths_SAM = entry_SAM[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################SAM##################################################################3
    ####################################################VLM#########################################
        entry_VLM = self.data_VLM[index]
        feat_VLM = np.load(os.path.join(self.root_VLM, entry_VLM[0] + '.npy'))

        feat_VLM = np.mean(feat_VLM, axis=1)


        labels_VLM = entry_VLM[1]

        hmap_VLM, num_action_VLM, center_loc_VLM, action_lengths_VLM = entry_VLM[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])

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
                labels_flow = labels_flow[random_index: random_index + num_clips: 1]
                hmap_flow = hmap_flow[random_index: random_index + num_clips: 1]

                features_depth = features_depth[random_index: random_index + num_clips: 1]
                labels_depth = labels_depth[random_index: random_index + num_clips: 1]
                hmap_depth = hmap_depth[random_index: random_index + num_clips: 1]

                features_pose = features_pose[random_index: random_index + num_clips: 1]
                labels_pose = labels_pose[random_index: random_index + num_clips: 1]
                hmap_pose = hmap_pose[random_index: random_index + num_clips: 1]

                features_SAM = features_SAM[random_index: random_index + num_clips: 1]
                labels_SAM = labels_SAM[random_index: random_index + num_clips: 1]
                hmap_SAM = hmap_SAM[random_index: random_index + num_clips: 1]

                features_VLM = features_VLM[random_index: random_index + num_clips: 1]
                labels_VLM = labels_VLM[random_index: random_index + num_clips: 1]
                hmap_VLM = hmap_VLM[random_index: random_index + num_clips: 1]
        else:

            features_flow = 0
            labels_flow = 0
            hmap_flow = 0

            features_depth = 0
            labels_depth = 0
            hmap_depth = 0

            features_pose = 0
            labels_pose = 0
            hmap_pose = 0

            features_SAM = 0
            labels_SAM = 0
            hmap_SAM = 0

            features_VLM = 0
            labels_VLM =0
            hmap_VLM = 0

        return features_rgb, labels_rgb, hmap_rgb, action_lengths_rgb, [entry_rgb[0], entry_rgb[2], num_action_rgb], features_flow, labels_flow, hmap_flow, action_lengths_flow, [entry_flow[0], entry_flow[2], num_action_flow], features_depth, labels_depth, hmap_depth, action_lengths_depth, [entry_depth[0], entry_depth[2], num_action_depth], features_pose, labels_pose, hmap_pose, action_lengths_pose, [entry_pose[0], entry_pose[2], num_action_pose], features_SAM, labels_SAM, hmap_SAM, action_lengths_SAM, [entry_SAM[0], entry_SAM[2], num_action_SAM], features_VLM, labels_VLM, hmap_VLM, action_lengths_VLM, [entry_VLM[0], entry_VLM[2], num_action_VLM]

    def __len__(self):
        return len(self.data_rgb)


class collate_fn_unisize():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        # max_len1 = int(self.num_clips)
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
            m1 = np.zeros((max_len), np.float32)
            l1 = np.zeros((max_len, b[6].shape[1]), np.float32)
            h1 = np.zeros((max_len, b[7].shape[1]), np.float32)
            f1[:b[5].shape[0]] = b[5]
            m1[:b[5].shape[0]] = 1
            l1[:b[5].shape[0], :] = b[6]
            h1[:b[5].shape[0], :] = b[7]

            f2 = np.zeros((max_len, b[10].shape[1], b[10].shape[2], b[10].shape[3]), np.float32)
            m2 = np.zeros((max_len), np.float32)
            l2 = np.zeros((max_len, b[11].shape[1]), np.float32)
            h2 = np.zeros((max_len, b[12].shape[1]), np.float32)
            f2[:b[10].shape[0]] = b[10]
            m2[:b[10].shape[0]] = 1
            l2[:b[10].shape[0], :] = b[11]
            h2[:b[10].shape[0], :] = b[12]

            f3 = np.zeros((max_len, b[15].shape[1], b[15].shape[2], b[15].shape[3]), np.float32)
            m3 = np.zeros((max_len), np.float32)
            l3 = np.zeros((max_len, b[16].shape[1]), np.float32)
            h3 = np.zeros((max_len, b[17].shape[1]), np.float32)
            f3[:b[15].shape[0]] = b[15]
            m3[:b[15].shape[0]] = 1
            l3[:b[15].shape[0], :] = b[16]
            h3[:b[15].shape[0], :] = b[17]

            f4 = np.zeros((max_len, b[20].shape[1], b[20].shape[2], b[20].shape[3]), np.float32)
            m4 = np.zeros((max_len), np.float32)
            l4 = np.zeros((max_len, b[21].shape[1]), np.float32)
            h4 = np.zeros((max_len, b[22].shape[1]), np.float32)
            f4[:b[20].shape[0]] = b[20]
            m4[:b[20].shape[0]] = 1
            l4[:b[20].shape[0], :] = b[21]
            h4[:b[20].shape[0], :] = b[22]

            f5 = np.zeros((max_len, b[25].shape[1], b[25].shape[2], b[25].shape[3]), np.float32)
            m5 = np.zeros((max_len), np.float32)
            l5 = np.zeros((max_len, b[26].shape[1]), np.float32)
            h5 = np.zeros((max_len, b[27].shape[1]), np.float32)
            f5[:b[25].shape[0]] = b[25]
            m5[:b[25].shape[0]] = 1
            l5[:b[25].shape[0], :] = b[26]
            h5[:b[25].shape[0], :] = b[27]

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), video_to_tensor(f1), torch.from_numpy(m1), torch.from_numpy(l1), b[9], torch.from_numpy(h1), video_to_tensor(f2), torch.from_numpy(m2), torch.from_numpy(l2), b[14], torch.from_numpy(h2), video_to_tensor(f3), torch.from_numpy(m3), torch.from_numpy(l3), b[19], torch.from_numpy(h3), video_to_tensor(f4), torch.from_numpy(m4), torch.from_numpy(l4), b[24], torch.from_numpy(h4), video_to_tensor(f5), torch.from_numpy(m5), torch.from_numpy(l5), b[29], torch.from_numpy(h5)])

        return default_collate(new_batch)
class collate_fn_unisize_eval():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize_eval(self, batch):
        # max_len = int(self.num_clips)
        # max_len1 = int(self.num_clips)
        new_batch = []

        for b in batch:
            max_len = b[0].shape[0]


            f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)
            m = np.zeros((max_len), np.float32)
            l = np.zeros((max_len, b[1].shape[1]), np.float32)
            h = np.zeros((max_len, b[2].shape[1]), np.float32)
            f[:b[0].shape[0]] = b[0]
            m[:b[0].shape[0]] = 1
            l[:b[0].shape[0], :] = b[1]
            h[:b[0].shape[0], :] = b[2]

            f1 = 0
            m1 = 0
            l1 = 0
            h1 = 0


            f2 = 0
            m2 = 0
            l2 = 0
            h2 = 0


            f3 = 0
            m3 = 0
            l3 = 0
            h3 = 0


            f4 = 0
            m4 = 0
            l4 = 0
            h4 = 0

            f5 = 0
            m5 = 0
            l5 = 0
            h5 = 0

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), f1, m1, l1, 0, h1, f2, m2, l2, 0, h2, f3, m3, l3, 0, h3, f4, m4, l4, 0, h4, f5, m5, l5, 0, h5])

        return default_collate(new_batch)

