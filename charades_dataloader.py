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


def make_dataset(split_file, split, root_rgb, root_flow, root_depth, num_classes=15):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset_rgb = []
    dataset_flow = []
    dataset_depth = []
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

        num_feat_rgb = fts_rgb.shape[0]
        num_feat_flow = fts_flow.shape[0]
        num_feat_depth = fts_depth.shape[0]

        label_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        label_flow = np.zeros((num_feat_flow, num_classes), np.float32)
        label_depth = np.zeros((num_feat_depth, num_classes), np.float32)

        #
        hmap_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        hmap_flow = np.zeros((num_feat_flow, num_classes), np.float32)
        hmap_depth = np.zeros((num_feat_depth, num_classes), np.float32)

        action_lengths = []
        center_loc = []
        num_action = 0

        fps_rgb = num_feat_rgb / data[vid]['duration']
        fps_flow = num_feat_flow / data[vid]['duration']
        fps_depth = num_feat_depth / data[vid]['duration']

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
                    action_duration = int((ann[2] - ann[1]) * fps_flow)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap_flow[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])
        #################flow##################################################################3
        #################flow##################################################################3
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
                    action_duration = int((ann[2] - ann[1]) * fps_depth)
                    radius = int(action_duration / gamma)
                    generate_gaussian(hmap_depth[:, class_], center, radius, tau, ku)
                    num_action = num_action + 1
                    center_loc.append([center, class_])
                    action_lengths.append([action_duration])
        #################flow##################################################################3



        dataset_rgb.append(
            (vid, label_rgb, data[vid]['duration'], [hmap_rgb, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        dataset_flow.append(
            (vid, label_flow, data[vid]['duration'], [hmap_flow, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        dataset_depth.append(
            (vid, label_depth, data[vid]['duration'],
             [hmap_depth, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        i += 1

    return dataset_rgb, dataset_flow, dataset_depth


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root_rgb, root_flow, root_depth, batch_size, classes, num_clips, skip):

        self.data_rgb, self.data_flow, self.data_depth= make_dataset(split_file, split, root_rgb, root_flow, root_depth, classes)

        self.split = split
        self.split_file = split_file
        self.batch_size = batch_size
        self.root_rgb = root_rgb
        self.root_flow = root_flow
        self.root_depth = root_depth
        self.in_mem = {}
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        ################3RGB#########################################
        entry_rgb = self.data_rgb[index]
        feat_rgb = np.load(os.path.join(self.root_rgb, entry_rgb[0] + '.npy'))

        feat_rgb = feat_rgb.reshape((feat_rgb.shape[0], 1, 1, feat_rgb.shape[-1]))
        features_rgb = feat_rgb.astype(np.float32)

        labels_rgb = entry_rgb[1]

        hmap_rgb, num_action_rgb, center_loc_rgb, action_lengths_rgb = entry_rgb[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips
        # random_index = random.choice(range(0, len(features_rgb) - num_clips))
        if self.split in ["training", "testing"]:
            if len(features_rgb) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = random.choice(range(0, len(features_rgb) - num_clips))
                else:
                    random_index = random.choice(range(0, len(features_rgb) - num_clips))
                features_rgb = features_rgb[random_index: random_index + num_clips: 1]
                labels_rgb = labels_rgb[random_index: random_index + num_clips: 1]
                hmap_rgb = hmap_rgb[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################RGB##################################################################3

    ####################################################flow#########################################
        entry_flow = self.data_flow[index]
        feat_flow = np.load(os.path.join(self.root_flow, entry_flow[0] + '.npy'))
        #only for vificlip
        # feat_flow =  np.mean(feat_flow, axis=1)
        feat_flow = feat_flow.reshape((feat_flow.shape[0], 1, 1, feat_flow.shape[-1]))
        features_flow = feat_flow.astype(np.float32)


        labels_flow = entry_flow[1]

        hmap_flow, num_action_flow, center_loc_flow, action_lengths_flow = entry_flow[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips


        if self.split in ["training", "testing"]:
            if len(features_flow) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = random.choice(range(0, len(features_flow) - num_clips))
                else:
                    random_index = random.choice(range(0, len(features_flow) - num_clips))
                features_flow = features_flow[random_index: random_index + num_clips: 1]
                labels_flow = labels_flow[random_index: random_index + num_clips: 1]
                hmap_flow = hmap_flow[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################flow##################################################################3
    ####################################################depth#########################################
        entry_depth = self.data_depth[index]
        feat_depth = np.load(os.path.join(self.root_depth, entry_depth[0] + '.npy'))
        feat_depth =  np.mean(feat_depth, axis=1)
        feat_depth = feat_depth.reshape((feat_depth.shape[0], 1, 1, feat_depth.shape[-1]))
        features_depth = feat_depth.astype(np.float32)

        labels_depth = entry_depth[1]

        hmap_depth, num_action_depth, center_loc_depth, action_lengths_depth = entry_depth[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips

        if self.split in ["training", "testing"]:
            if len(features_depth) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = random.choice(range(0, len(features_depth) - num_clips))
                else:
                    random_index = random.choice(range(0, len(features_depth) - num_clips))
                features_depth = features_depth[random_index: random_index + num_clips: 1]
                labels_depth = labels_depth[random_index: random_index + num_clips: 1]
                hmap_depth = hmap_depth[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################flow##################################################################3

        return features_rgb, labels_rgb, hmap_rgb, action_lengths_rgb, [entry_rgb[0], entry_rgb[2], num_action_rgb], features_flow, labels_flow, hmap_flow, action_lengths_flow, [entry_flow[0], entry_flow[2], num_action_flow], features_depth, labels_depth, hmap_depth, action_lengths_depth, [entry_depth[0], entry_depth[2], num_action_depth]

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

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), video_to_tensor(f1), torch.from_numpy(m1), torch.from_numpy(l1), b[9], torch.from_numpy(h1), video_to_tensor(f2), torch.from_numpy(m2), torch.from_numpy(l2), b[14], torch.from_numpy(h2)])

        return default_collate(new_batch)