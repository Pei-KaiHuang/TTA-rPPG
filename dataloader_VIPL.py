from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F

import math   
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from PIL import Image
from torchvision import datasets, transforms
# from util import visualize_result_norm
import glob
import random
import threading

START_OFFSET=30


# def get_frame(paths, transform_face, gray=False):
#     face_frames = []
#     for path in paths:
#         frame =  Image.open(path)
#         if gray:
#             frame = frame.convert('L')
            
#         face_frame = transform_face(frame)
#         face_frames.append(face_frame)
#     return face_frames


def process_images(paths, transform_face, gray, results, index):
    """Function that each thread will execute."""
    frames = []
    for path in paths:
        frame = Image.open(path)
        if gray:
            frame = frame.convert('L')
        processed_frame = transform_face(frame)
        frames.append(processed_frame)
    results[index] = frames



def get_frame(paths, transform_face, gray=False, num_threads=8):
    """Loads and processes frames using multiple threads."""
    thread_list = []
    results = [None] * num_threads  # List to store results from each thread
    n = len(paths)
    images_per_thread = n // num_threads

    for i in range(num_threads):
        start_index = i * images_per_thread
        end_index = start_index + images_per_thread
        if i == num_threads - 1:  # Handle the last batch
            end_index = n
        thread = threading.Thread(target=process_images, args=(paths[start_index:end_index], transform_face, gray, results, i))
        thread_list.append(thread)
        thread.start()

    for thread in thread_list:
        thread.join()

    # Combine the results from all threads
    all_frames = []
    for frames in results:
        all_frames.extend(frames)

    return all_frames



def get_rPPG(path):

    f = open(path, 'r')
    lines = f.readlines()
    PPG = [float(ppg) for ppg in lines[0].split()]
    # hr = [float(ppg) for ppg in lines[1].split()[:100]]
    # no = [float(ppg) for ppg in lines[2].split()[:100]]
    f.close()

    return PPG


class rPPG_Dataset(Dataset):
    def __init__(self, datasets, seq_length, train, if_bg=True, num_batch_per_sample=1, testFold=1):
        
        self.train = train
        self.if_bg = if_bg
        self.num_batch_per_sample = num_batch_per_sample
        
        face_folder = "crop"
        self.root_dir = f"../dataset/VIPL-HR/VIPL-HR_{face_folder}"
                
        bg_folder = "MiDaS" # "bg" or "naive_masked"
        self.root_bg_dir = f"../dataset/rPPG_dataset/VIPL-HR/VIPL-HR_{bg_folder}"
        
        
        self.datasets = datasets
        self.subjects = {}
        self.subjects["V"] = []
        
        self.subject_images = {}
        self.bg_images = {}
        
        self.subject_GT_path = {}
        self.subject_GT_PPG = {}
        
        allFolds = [1,2,3,4,5]
        prefix = "preprocess_data_fold"
        
        if testFold != 0:
            assert testFold in allFolds

            if train:
                allFolds.remove(testFold)
            else:
                allFolds = [testFold]
        
        
        self.subjects["V"] = []
        print("Training fold:", allFolds)
        
        for fold in allFolds:
            
            subjects = os.listdir(f"{self.root_dir}/{prefix}{fold}")
            subjects = [f"{prefix}{fold}/{s}" for s in subjects if os.path.isdir(f"{self.root_dir}/{prefix}{fold}/{s}")]
            
            # TODO: missing ground truth, temporarily remove it
            if fold==1:
                subjects.remove("preprocess_data_fold1/p49_v2_source1")
            
            self.subjects["V"].extend(subjects)
        


        for _subject in self.subjects["V"]:
            
            image_paths = sorted(glob.glob(f"{self.root_dir}/{_subject}/*.png"))
            bg_paths = sorted(glob.glob(f"{self.root_bg_dir}/{_subject}/*.png"))

            _key = f"V_{_subject}"
            self.subject_images[_key] = image_paths
            self.bg_images[_key] = bg_paths
            
            ground_truth = os.path.join("../dataset/VIPL-HR/GTs", _subject.split('/')[-1], "ground_truth4.txt")
            self.subject_GT_path[_key] = ground_truth
            self.subject_GT_PPG[_key] = get_rPPG(ground_truth)
                
            # if(len(image_paths) != len(self.subject_GT_PPG[_key])):
            #     print(_key, len(image_paths), len(self.subject_GT_PPG[_key]))
        
        self.all_keys = list(self.subject_images.keys())
        self.seq_length = seq_length
        print(f"Total number of samples: {len(self.all_keys)}")

        self.transform_face = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        

    def __getitem__(self, idx):
        

        """Returns a dataset item given an index."""
        _key = self.all_keys[idx]
        _face_frame, _bg_frame, _ppg = [], [], []
        
        start = START_OFFSET if not self.train else random.randint(START_OFFSET, 
                                                                   max(START_OFFSET, 
                                                                       len(self.subject_images[_key]) - self.seq_length)
                                                                   )
        
        total_frame = len(self.subject_images[_key])
        num_batch = min(max(total_frame // self.seq_length, 1), self.num_batch_per_sample)
        # num_batch = self.num_batch_per_sample
        
        _face_frame_paths = self.subject_images[_key][start:start+self.seq_length*num_batch]   
        _face_frame = get_frame(_face_frame_paths, self.transform_face)
        _face_frame = torch.stack(_face_frame).transpose(0, 1)
        
        if self.if_bg:
            _bg_frame_paths = self.bg_images[_key][start:start+self.seq_length*num_batch]   
            _bg_frame = get_frame(_bg_frame_paths, self.transform_face)
            _bg_frame = torch.stack(_bg_frame).transpose(0, 1)
        
        _ppg = torch.FloatTensor(self.subject_GT_PPG[_key][start:start+self.seq_length*num_batch])
        
        
        if _face_frame.shape[1] < self.seq_length*num_batch:
            _face_frame = F.pad(_face_frame, (0, 0, 0, 0, 0, self.seq_length*num_batch-_face_frame.shape[1]))
            _ppg = F.pad(_ppg, (0, self.seq_length*num_batch-_ppg.shape[0]))
            
            
        if self.if_bg and _bg_frame.shape[1] < self.seq_length*num_batch:
            _bg_frame = F.pad(_bg_frame, (0, 0, 0, 0, 0, self.seq_length-_bg_frame.shape[1]))
                            

        return _face_frame, _bg_frame, _ppg, _key, num_batch


    def __len__(self):
        # return 10
        return len(self.all_keys)
    


    
def get_loader(_datasets, _seq_length, batch_size=1, shuffle=True, train=True, if_bg=True, testFold=1, num_batch_per_sample=1):
    
    _dataset = rPPG_Dataset(datasets=_datasets, 
                            seq_length=_seq_length,
                            train=train,
                            if_bg=if_bg,
                            testFold=testFold,
                            num_batch_per_sample=num_batch_per_sample)

    # num_id, num_domain = _dataset.get_id_domain_num()
    
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":
    
    train_loader = get_loader(_datasets=list("V"),
                                _seq_length=300,
                                batch_size=1,
                                testFold=1,
                                train=False,
                                if_bg=False,
                                num_batch_per_sample=3,)
    

    for step, (face_frames, bg_frames, ppg_labels, subjects, num_batch) in enumerate(train_loader):
        
        print(f"{face_frames.shape=}, {ppg_labels.shape=}, {subjects=}, {num_batch=}")
        # break