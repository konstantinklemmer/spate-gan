import os
import json
import sys
import requests
from urllib.request import urlretrieve
import urllib.request
import zipfile
import subprocess
import argparse
import time
from datetime import datetime

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

class MyDataset(IterableDataset):

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

def download_file_from_google_drive(id, destination):
    URL = "http://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def fetch_extreme_weather(time_steps=10, small=False):
    if small==False:
      file_id = '1GifNUwnCvxcRuiUdCFWq2eN8ePNrNoZf'
    else:
      file_id = '15ecYVzd3IeZSRO89IH81rFgQzuUmu1H0'
    destination = './surf_temp_rad_2004.npy'
    download_file_from_google_drive(file_id, destination)
    # Load from local environment
    td = np.load('./surf_temp_rad_2004.npy')
    td = torch.from_numpy(td)
    # Prepare data (same as done for the point process data)
    total_time_steps, nc, x_height, x_width = td.shape
    data_seq = torch.reshape(td, (total_time_steps // time_steps, time_steps, x_height, x_width))
    data_seq = data_seq.permute(2, 3, 0, 1)
    raw_data = data_seq.permute(0, 2, 1, 3).permute(1, 0, 2, 3).permute(0, 1, 3, 2).permute(0, 2, 1, 3)
    raw_data = raw_data.reshape(raw_data.shape[0], raw_data.shape[1], 1, raw_data.shape[2], raw_data.shape[3])
    # normalise data between 0 and 1
    data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data)) 
    dataset = MyDataset(data)
    return dataset, x_height, x_width
    
def fetch_lgcp(time_steps=10,d_type="gneiting_3000"):
    if d_type=="gneiting_1000":
      file_id = '16MkWedpy9MNukTk00kxYJTaStpOQ9q2M'
    if d_type=="gneiting_3000":
      file_id = '1xeX5rXCXsrVH-UflZghBnsNSNwS6IZF7'
    if d_type=="exponential_1000":
      file_id = '1HgZFK4URKXsPWaYKFkp4mmifQenCrQT6'
    destination = './lgcp.json'
    download_file_from_google_drive(file_id, destination)
    # Load from local environment
    with open('./lgcp.json') as json_file:
        data = np.array(json.load(json_file)).astype(np.float32)
    data = torch.tensor(data)
    x_height, x_width, total_time_steps = data.shape
    data_seq = torch.reshape(data, (x_height, x_width, total_time_steps // time_steps, time_steps))
    raw_data = data_seq.permute(0, 2, 1, 3).permute(1, 0, 2, 3).permute(0, 1, 3, 2).permute(0, 2, 1, 3)
    # normalise data between 0 and 1
    data = (raw_data - torch.min(raw_data)) / (torch.max(raw_data) - torch.min(raw_data))
    data = data.reshape(total_time_steps // time_steps, time_steps, 1, x_height, x_width)
    dataset = MyDataset(data)
    return dataset, x_height, x_width

def fetch_turbulent_flows(scaled_h=64, scaled_w=64,single_chann=True): # This dataset has predefined time_steps = 7
    subprocess.check_output(['gdown', '--id', '19KWD9vIj1BjiEa5xcsHWA2kDvA2_bvLU'])
    data = torch.load("rbc_data.pt")
    n, nc, h, wt = data.shape
    t = 7
    w = wt // t
    data = data.reshape(n, nc, h, t, w)
    data = data.permute(0, 3, 1, 2, 4)
    data = data.permute(0, 1, 3, 4, 2)
    data = np.asarray(data)

    down_data = np.empty((n, t, scaled_h, scaled_w, nc), np.float32)

    for i in range(n):
        for j in range(t):
            down_data[i, j, ...] = cv2.resize(data[i, j, ...], dsize=(scaled_h, scaled_w))

    data = torch.from_numpy(down_data)
    data = data.permute(0, 1, 4, 2, 3)
    if single_chann==True:
      data = data[:, :, :1, :, :]
    # torch.save(down_data, "../data/weather/downscaled_rbc_data.pt")
    data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    dataset = MyDataset(data)
    return dataset, scaled_h, scaled_w