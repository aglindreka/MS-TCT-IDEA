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


def make_dataset(split_file, split, root_rgb, root_flow, num_classes=157):
    gamma = 0.5
    tau = 4
    ku = 1
    dataset_rgb = []
    dataset_flow = []
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
        num_feat_rgb = fts_rgb.shape[0]
        num_feat_flow = fts_flow.shape[0]

        label_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        label_flow = np.zeros((num_feat_flow, num_classes), np.float32)
        #
        hmap_rgb = np.zeros((num_feat_rgb, num_classes), np.float32)
        hmap_flow = np.zeros((num_feat_flow, num_classes), np.float32)

        action_lengths = []
        center_loc = []
        num_action = 0

        fps_rgb = num_feat_rgb / data[vid]['duration']
        fps_flow = num_feat_flow / data[vid]['duration']
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



        dataset_rgb.append(
            (vid, label_rgb, data[vid]['duration'], [hmap_rgb, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        dataset_flow.append(
            (vid, label_flow, data[vid]['duration'], [hmap_flow, num_action, np.asarray(center_loc), np.asarray(action_lengths)]))
        i += 1

    return dataset_rgb, dataset_flow


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root_rgb, root_flow, batch_size, classes, num_clips, skip):

        self.data = make_dataset(split_file, split, root_rgb, root_flow, classes)
        self.split = split
        self.split_file = split_file
        self.batch_size = batch_size
        self.root_rgb = root_rgb
        self.root_flow = root_flow
        self.in_mem = {}
        self.num_clips = num_clips
        self.skip = skip

    def __getitem__(self, index):
        ################3RGB#########################################
        entry_rgb = self.data[0][index]
        feat_rgb = np.load(os.path.join(self.root_rgb, entry_rgb[0] + '.npy'))
        feat_rgb = feat_rgb.reshape((feat_rgb.shape[0], 1, 1, feat_rgb.shape[-1]))
        features_rgb = feat_rgb.astype(np.float32)

        labels_rgb = entry_rgb[1]

        hmap_rgb, num_action_rgb, center_loc_rgb, action_lengths_rgb = entry_rgb[3]
        # print('center_loc',center_loc.shape)
        # center_loc = np.transpose(center_loc, axes=[1, 0])
        num_clips = self.num_clips

        if self.split in ["training", "testing"]:
            if len(features_rgb) > num_clips and num_clips > 0:
                if self.split == "testing":
                    random_index = 0
                else:
                    random_index = random.choice(range(0, len(features_rgb) - num_clips))
                features_rgb = features_rgb[random_index: random_index + num_clips: 1]
                labels_rgb = labels_rgb[random_index: random_index + num_clips: 1]
                hmap_rgb = hmap_rgb[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################RGB##################################################################3

        ################3RGB#########################################
        entry_flow = self.data[1][index]
        feat_flow = np.load(os.path.join(self.root_flow, entry_flow[0] + '.npy'))
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
                    random_index = 0
                else:
                    random_index = random.choice(range(0, len(features_flow) - num_clips))
                features_flow = features_flow[random_index: random_index + num_clips: 1]
                labels_flow = labels_flow[random_index: random_index + num_clips: 1]
                hmap_flow = hmap_flow[random_index: random_index + num_clips: 1]
        # center_loc = np.transpose(center_loc, axes=[1, 0])
    ##########################RGB##################################################################3

        return features_rgb, labels_rgb, hmap_rgb, action_lengths_rgb, [entry_rgb[0], entry_rgb[2], num_action_rgb], features_flow, labels_flow, hmap_flow, action_lengths_flow, [entry_flow[0], entry_flow[2], num_action_flow]

    def __len__(self):
        return len(self.data)


class collate_fn_unisize():

    def __init__(self, num_clips):
        self.num_clips = num_clips

    def charades_collate_fn_unisize(self, batch):
        max_len = int(self.num_clips)
        max_len1 = int(self.num_clips)
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

            new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[4], torch.from_numpy(h), video_to_tensor(f1), torch.from_numpy(m1), torch.from_numpy(l1), b[9], torch.from_numpy(h1)])

        return default_collate(new_batch)