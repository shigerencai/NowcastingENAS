import os
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from default_configs.default_config import DefaultConfig


def clean_fold(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)


def pixel_to_dBZ(img):
    if np.max(img) > 1 and np.min(img) >= 0:
        img = img/255.0
    img = img * 95.0 -10
    img = img.astype(np.int64)
    img[img < -10] = -10
    return img

def dBZ_to_pixel(dBZ_img):
    return (dBZ_img.astype(np.float32) + 10.0) * 255.0 / 95.0


def padding_CIKM_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros((batch_size, seq_length, DefaultConfig.get_config('CIKM_img_width'), DefaultConfig.get_config('CIKM_img_height'), DefaultConfig.get_config('CIKM_img_channel')))
    padding_frame_dat[:, :, 13:-14, 13:-14, :] = frame_data
    return padding_frame_dat


def unpadding_CIKM_data(padding_frame_dat):
    return padding_frame_dat[:, :, 13:-14, 13:-14, :]

def padding_Shanghai_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros((batch_size, seq_length, DefaultConfig.get_config('Shanghai_img_width'), DefaultConfig.get_config('Shanghai_img_height'), DefaultConfig.get_config('Shanghai_img_channel')))
    padding_frame_dat[:, :, 5:-6, 5:-6, :] = frame_data
    return padding_frame_dat

def unpadding_Shanghai_data(padding_frame_dat):
    return padding_frame_dat[:, :, 5:-6, 5:-6, :]

def schedule_CIKM_sampling(eta, itr):
    zeros = np.zeros((DefaultConfig.get_config('batch_size'),
                      DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1,
                      DefaultConfig.get_config('CIKM_img_width') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('CIKM_img_height') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('CIKM_img_channel')))
    if not DefaultConfig.get_config('scheduled_sampling'):
        return 0.0, zeros

    if itr < DefaultConfig.get_config('sampling_stop_iter'):
        eta -= DefaultConfig.get_config('sampling_changing_rate')
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (DefaultConfig.get_config('batch_size'), DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1))
    true_token = (random_flip < eta)
    ones = np.ones((DefaultConfig.get_config('CIKM_img_width') // DefaultConfig.get_config('patch_size'),
                    DefaultConfig.get_config('CIKM_img_height') // DefaultConfig.get_config('patch_size'),
                    DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('CIKM_img_channel')))
    zeros = np.zeros((DefaultConfig.get_config('CIKM_img_width') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('CIKM_img_height') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('CIKM_img_channel')))
    real_input_flag = []
    for i in range(DefaultConfig.get_config('batch_size')):
        for j in range(DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (DefaultConfig.get_config('batch_size'),
                            DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1,
                            DefaultConfig.get_config('CIKM_img_width') // DefaultConfig.get_config('patch_size'),
                            DefaultConfig.get_config('CIKM_img_height') // DefaultConfig.get_config('patch_size'),
                            DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('CIKM_img_channel')))

    return eta, real_input_flag

def schedule_Shanghai_sampling(eta, itr):
    zeros = np.zeros((DefaultConfig.get_config('batch_size'),
                      DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1,
                      DefaultConfig.get_config('Shanghai_img_width') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('Shanghai_img_height') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('Shanghai_img_channel')))
    if not DefaultConfig.get_config('scheduled_sampling'):
        return 0.0, zeros

    if itr < DefaultConfig.get_config('sampling_stop_iter'):
        eta -= DefaultConfig.get_config('sampling_changing_rate')
    else:
        eta = 0.0

    random_flip = np.random.random_sample(
        (DefaultConfig.get_config('batch_size'), DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1))
    true_token = (random_flip < eta)
    ones = np.ones((DefaultConfig.get_config('Shanghai_img_width') // DefaultConfig.get_config('patch_size'),
                    DefaultConfig.get_config('Shanghai_img_height') // DefaultConfig.get_config('patch_size'),
                    DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('Shanghai_img_channel')))
    zeros = np.zeros((DefaultConfig.get_config('Shanghai_img_width') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('Shanghai_img_height') // DefaultConfig.get_config('patch_size'),
                      DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('Shanghai_img_channel')))
    real_input_flag = []
    for i in range(DefaultConfig.get_config('batch_size')):
        for j in range(DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (DefaultConfig.get_config('batch_size'),
                            DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1,
                            DefaultConfig.get_config('Shanghai_img_width') // DefaultConfig.get_config('patch_size'),
                            DefaultConfig.get_config('Shanghai_img_height') // DefaultConfig.get_config('patch_size'),
                            DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('Shanghai_img_channel')))
    return eta, real_input_flag

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    patch_tensor = img_tensor
    if patch_size != 1:
        batch_size = np.shape(img_tensor)[0]
        seq_length = np.shape(img_tensor)[1]
        img_height = np.shape(img_tensor)[2]
        img_width = np.shape(img_tensor)[3]
        num_channels = np.shape(img_tensor)[4]
        a = np.reshape(img_tensor, [batch_size, seq_length,
                                    img_height//patch_size, patch_size,
                                    img_width//patch_size, patch_size,
                                    num_channels])
        b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
        patch_tensor = np.reshape(b, [batch_size, seq_length,
                                      img_height//patch_size,
                                      img_width//patch_size,
                                      patch_size*patch_size*num_channels])
    return patch_tensor



def reshape_patch_back(patch_tensor, patch_size):
    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape
    img_tensor = patch_tensor
    if patch_size != 1:
        img_channels = channels // (patch_size * patch_size)
        if 'numpy' in str(type(patch_tensor)):
            a = patch_tensor.reshape(batch_size, seq_length, patch_height, patch_width, patch_size, patch_size, img_channels)
            b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
            img_tensor = b.reshape(batch_size, seq_length, patch_height*patch_size, patch_width*patch_size, img_channels)
        else:
            a = patch_tensor.reshape(batch_size, seq_length, patch_height, patch_width, patch_size, patch_size, img_channels)
            b = a.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
            img_tensor = b.reshape(batch_size, seq_length, patch_height*patch_size, patch_width*patch_size, img_channels)

    return img_tensor



def nor(frames):
    new_frames = frames.astype(np.float32)/255.0
    return new_frames



def de_nor(frames):
    new_frames = copy.deepcopy(frames)
    new_frames *= 255.0
    new_frames = new_frames.astype(np.uint8)
    return new_frames


def normalization(frames,up=80):
    new_frames = frames.astype(np.float32)
    new_frames /= (up/2)
    new_frames -= 1
    return new_frames


def denormalization(frames,up=80):
    new_frames = copy.deepcopy(frames)
    new_frames += 1
    new_frames *= (up/2)
    new_frames = new_frames.astype(np.uint8)
    return new_frames



