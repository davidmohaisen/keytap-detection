import json
import numpy as np
import pickle
import pprint
import os, os.path, sys
import bisect
from functools import reduce


from pathlib import Path

import glob, os

import random

import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from sklearn import metrics

from model import Net

import matplotlib.pyplot as plt

from utility import *

whole_results = { 'left': { 'before_refining': {}, 'after_refining': {} }, 'right': { 'before_refining': {}, 'after_refining': {} } }
rec_fold = None

#####################################################################################################
# CONFIG
#####################################################################################################
# No overfit, reg-5
left_model_path = "models/left_model_reg10e-1_lr10e-4-500.pt"
right_model_path = "models/right_model_reg10e-1_lr10e-4-1000.pt"

#left_model_path = "models/new_data_left_model_reg10e-1_lr10e-4.pt"
#right_model_path = "models/new_data_right_model_reg10e-1_lr10e-4.pt"


# input: filepath
# output: list of strings, each delimited with tab character in the input file
def read_data_from_tab_delimited_file(path):
    with open(path) as f:
        line = f.readline()
        line = line.replace("\n","").split("\t")
        return line

def getTime(a):
    a = a.replace("(", "")
    a = a.replace(")", "")
    a = a.split(",")
    return int(a[-1])

# def getFinger(a):
#     if a.find("index") > 0:
#         return "index"
#     elif a.find("middle") > 0:
#         return "middle"

def getHand(a):
    if a.find("left") > 0:
        return "left"
    elif a.find("right") > 0:
        return "right"
    else:
        assert False

# returns indices, hands, fingers, all as lists ([int], [str], [str])
def read_labels(filepath):
    with open(filepath) as f:
        line = f.readline()
        line = line.replace("\n","")
        line = line.split("\t")[:-1:]
        
        left_keytap_indices = []
        right_keytap_indices = []

        for tup in line:
            hand = getHand(tup)

            if hand == "left":
                left_keytap_indices.append( getTime(tup) )
            elif hand == "right":
                right_keytap_indices.append( getTime(tup) )
            else:
                assert False

        return left_keytap_indices, right_keytap_indices

###############################################################################

# path, left/right, finger name, feature name
# example: "C:/blabla/data/", "left", "thumb", "pos_x"
def get_full_filepath(path, which_hand, which_finger, which_feature):
    suffix = which_hand.lower() + "_" + which_finger.upper() + "_" + which_feature.upper() + ".txt"
    return os.path.join(path, suffix)

# list of proposal frame windows as (start, end)
def get_proposal_frame_windows(mintime, maxtime, window_len, step):
    windows = []

    start = mintime

    while start + window_len < maxtime:
        windows.append( (start, start + window_len) )
        start += step

    return windows

def get_IoU_for_windows(windows, keytap_moments, window_len, is_window=True):
    mid_points_for_windows = [(start+end)//2 for start,end in windows] if is_window else windows

    IoU = []

    if not keytap_moments:
        return [0] * len(windows)

    # for each window, find the overlap with each keytap moment
    for mid_point in mid_points_for_windows:
        # the distance btw the midpoint with each keytap moment
        distances = [ abs(mid_point - keytap_moment) for keytap_moment in keytap_moments ]
        
        # find the closest
        min_dist = min(distances)

        # compute the IoU
        intersection = 0 if min_dist > (window_len / 2) else ((window_len / 2) - min_dist) / (window_len / 2)

        IoU.append(intersection)
    
    return IoU

# returns data_dict and keytap_moments
def parse_data_from_folder(path):
    """
    # ang           : angle between bone2 and bone3
    # dir_x|y|z     : the direction of the tip of the finger (where it points at)
    # pos_x|y|z     : the tip position of the finger wrt the palm of the hand
    # rel_pos_x|y|z : the relative position of each finger with respect to the finger next to it
    """
    hand_names = "left", "right"
    finger_names = "thumb", "index", "middle", "ring", "pinky"
    feature_names = "ang", "dir_x", "dir_y", "dir_z", "mov_dir_x", "mov_dir_y", "mov_dir_z", "pos_x", "pos_y", "pos_z", "rel_pos_x", "rel_pos_y", "rel_pos_z"

    # fill data_dict such that data[hand][finger][feature] would give a list of floats
    data_dict = {}

    # read all data
    for hand in hand_names:
        data_dict[hand] = {}

        for finger in finger_names:
            data_dict[hand][finger] = {}

            for feature in feature_names:
                filepath = get_full_filepath(path, hand, finger, feature)
                data_string_list = read_data_from_tab_delimited_file(filepath)
                data_float_list = list(map(float, data_string_list))

                data_dict[hand][finger][feature] = data_float_list # TODO: This was np.array
    # read labels
    labels_filepath = os.path.join(path, "labels.txt")
    left_keytap_moments, right_keytap_moments = read_labels(labels_filepath)


    ''' Difference between two consecutive keytaps
    diffs = []
    for i in range(len(keytap_moments) - 1):
        diffs.append( keytap_moments[i+1] - keytap_moments[i] )
    
    #diffs.sort()
    print(diffs)
    exit()
    '''

    frame_count = len(data_dict[hand_names[0]][finger_names[0]][feature_names[0]])

    return data_dict, left_keytap_moments, right_keytap_moments, frame_count

def save_parsed_data_as_json_file(data_dict, left_keytap_moments, right_keytap_moments, frame_count, path):
    data = { "data" : data_dict, "left_keytap_moments": left_keytap_moments, "right_keytap_moments": right_keytap_moments, "frame_count" : frame_count }

    with open(path, 'w') as outfile:
        json.dump(data, outfile)

# returns data, keytap_moments, frame_count
def load_parsed_data_from_json_file(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    
    return data["data"], data["left_keytap_moments"], data["right_keytap_moments"], data["frame_count"]

""" 
# return: windows (start, end) and IoUs (0-1)
# if binary_data is set to True, only returns windows with IoU 0 or 1, nothing else, which should be used for training
# otherwise, returns windows that are step away from each other with uniformly distributed number of zero, non-zero, one values
"""
def get_uniformly_distributed_windows(keytap_moments, frame_count, window_len, start, end, step, binary_data):
    windows = get_proposal_frame_windows(start, end, window_len, step) # all possible windows within given start, end with window_len
    IoU = get_IoU_for_windows(windows, keytap_moments, window_len)     # the highest IoU of each window with a ground truth keytap moment

    zeros = []
    ones = []
    last_zero_start = 0

    # binary classification - only 0s and 1s - do not consider the ones with IoU that are not 0 or 1
    # this one is used while training the model
    if binary_data:
        # get ones and zeros
        for window, IoU in zip(windows, IoU):
            if IoU == 0 and (last_zero_start + window_len) <= window[0]:
                zeros.append( (window, IoU) )
                last_zero_start = window[0]
            elif IoU == 1:
                ones.append( (window, IoU) )
        
        data = ones + zeros

        # get windows
        windows = [d[0] for d in data]

        # get IoU for each window
        IoU = [d[1] for d in data]

        return windows, IoU

        # randomly sample from the one with higher count - most possibly zeros
        # zeros = random.sample(zeros, min(len(zeros), len(ones) ) )
        # ones = random.sample(ones, min(len(zeros), len(ones) ) )

    else: # if it is test, then we should test everything since in a real case we wont know about the labels
        return windows, IoU
        # for window, IoU in zip(windows, IoU):
        #     if IoU == 0 and (last_zero_start + window_len) <= window[0]:
        #         zeros.append( (window, IoU) )
        #         last_zero_start = window[0]
        #     elif IoU != 0 and IoU != 1:
        #         non_zeros_non_ones.append( (window, IoU) )
        #     elif IoU == 1:
        #         ones.append( (window, IoU) )

        # typically, non_zeros_non_ones have two times count then ones
        # zeros = random.sample(zeros, min(len(zeros), len(ones) ) )
        # ones = random.sample(ones, min(len(zeros), len(ones) ) )
        # non_zeros_non_ones = random.sample(non_zeros_non_ones, min(len(zeros), len(ones) ) )

    # To learn the number of unique IoU values
    #unique_IoU_vals = sorted(list(set(IoU)))
    #num = []
    #for u in unique_IoU_vals:
    #    num.append( (u, sum(list(map( lambda x: x==u, IoU )))))
    # compute weights


# returns feature matrices, sliced as windows. list of feature matrices: [feature1, feature2 .. featuren], each numpy array
# shape for each =  number of windows, depth-xyz(3), num of fingers(5), window_len(10)
def get_input_features(data_dict, windows, window_len, hand_name):
    # get features and put each in a numpy array
    feature_1_stacked = {} # moving direction
    feature_2_stacked = {} # position
    feature_3_stacked = {} # direction

    # from windows, create instances of mov_dir data, each of which with dim: height=5(fingers), width=window_len, depth=3(xyz)
    feature1_names = "mov_dir_x", "mov_dir_y", "mov_dir_z"
    feature2_names = "pos_x", "pos_y", "pos_z"
    feature3_names = "dir_x", "dir_y", "dir_z"
    
    # for each finger, stack the dimension of each finger at depth dimension
    finger_names = "thumb", "index", "middle", "ring", "pinky"
    for finger in finger_names:
        feature1_stacked_fingers = []
        feature2_stacked_fingers = []
        feature3_stacked_fingers = []

        # feature 1
        for feature in feature1_names:
            data = data_dict[hand_name][finger][feature]
            feature1_stacked_fingers.append(data)

        # feature 2
        for feature in feature2_names:
            data = data_dict[hand_name][finger][feature]
            feature2_stacked_fingers.append(data)

        # feature 3
        for feature in feature3_names:
            data = data_dict[hand_name][finger][feature]
            feature3_stacked_fingers.append(data)

        # stack feature 1
        feature1_np_stacked = np.stack( feature1_stacked_fingers )
        feature_1_stacked[finger] = feature1_np_stacked.transpose()

        # stack feature 2
        feature2_np_stacked = np.stack( feature2_stacked_fingers )
        feature_2_stacked[finger] = feature2_np_stacked.transpose()

        # stack feature 3
        feature3_np_stacked = np.stack( feature3_stacked_fingers )
        feature_3_stacked[finger] = feature3_np_stacked.transpose()

    # stack fingers at height
    # [5, 3, instance_count], where 5: finger count, 3: xyz
    fingers_feature1_allstacked = np.stack( \
        [ feature_1_stacked["thumb"], \
        feature_1_stacked["index"], \
        feature_1_stacked["middle"], \
        feature_1_stacked["ring"], \
        feature_1_stacked["pinky"] ] )

    fingers_feature2_allstacked = np.stack( \
        [ feature_2_stacked["thumb"], \
        feature_2_stacked["index"], \
        feature_2_stacked["middle"], \
        feature_2_stacked["ring"], \
        feature_2_stacked["pinky"] ] )

    fingers_feature3_allstacked = np.stack( \
        [ feature_3_stacked["thumb"], \
        feature_3_stacked["index"], \
        feature_3_stacked["middle"], \
        feature_3_stacked["ring"], \
        feature_3_stacked["pinky"] ] )

    # create the feature matrices for each window
    feature1_of_each_keytap_frame = []
    feature2_of_each_keytap_frame = []
    feature3_of_each_keytap_frame = []

    for window in windows:
        # feature1
        one_slice = fingers_feature1_allstacked[ : , window[0] : window[1] , : ]
        feature1_of_each_keytap_frame.append( one_slice )

        # feature2
        one_slice = fingers_feature2_allstacked[ : , window[0] : window[1] , : ]
        feature2_of_each_keytap_frame.append( one_slice )

        # feature3
        one_slice = fingers_feature3_allstacked[ : , window[0] : window[1] , : ]
        feature3_of_each_keytap_frame.append( one_slice )

    # reshape the feature matrices
    feature1_of_each_keytap_frame = np.array( feature1_of_each_keytap_frame ).reshape( len(windows), 3, 5, window_len ) # 3: depth (xyz), 5: num of fingers
    feature2_of_each_keytap_frame = np.array( feature2_of_each_keytap_frame ).reshape( len(windows), 3, 5, window_len )
    feature3_of_each_keytap_frame = np.array( feature3_of_each_keytap_frame ).reshape( len(windows), 3, 5, window_len )

    # stack features at the newly created last axis to be resolved at the model
    features = np.concatenate( \
        [feature1_of_each_keytap_frame[..., np.newaxis],\
         feature2_of_each_keytap_frame[..., np.newaxis],\
         feature3_of_each_keytap_frame[..., np.newaxis]], axis=-1 )

    return features

# The function includes usage of some hyper-parameters - be careful
# data distribution should be either test or train. train produces binary data (see get_uniformly_distributed_windows())
def process_and_get_np(folder, data_distribution, hand_name):
    json_file_path = os.path.join(folder, "data_json_file.json")

    # Check if json file already exists
    if os.path.exists(json_file_path):
        print("process_and_get_np(): JSON file exists. Reading from JSON file..")
        data_dict, left_keytap_moments, right_keytap_moments, frame_count = load_parsed_data_from_json_file(json_file_path)
    else:
        # Read data from folder
        print("process_and_get_np(): JSON file doesn't exist. Reading from each txt and saving to JSON file..")
        data_dict, left_keytap_moments, right_keytap_moments, frame_count = parse_data_from_folder(folder)

        save_parsed_data_as_json_file(data_dict, left_keytap_moments, right_keytap_moments, frame_count, json_file_path )

    binary_data = data_distribution.lower() == "train"
    window_len = 30

    # Now, continue with the keytap moments of the desired hand
    if hand_name == 'left':
        keytap_moments = left_keytap_moments
    elif hand_name == 'right':
        keytap_moments = right_keytap_moments
    else:
        assert False

    # windows and the IoU corresponding to each
    windows, IoU = get_uniformly_distributed_windows(keytap_moments, frame_count, window_len=window_len, start=0, end=frame_count, step=1, binary_data=binary_data)

    # process and get input features
    features = get_input_features(data_dict, windows, window_len, hand_name) # dim0: features, dim1: instances, dim2: xyz, dim3: fingers, dim4: window len

    # return
    return np.array(features), np.array(IoU), windows, IoU, keytap_moments


###############################################################################################################
# Train & Evaluate
###############################################################################################################

def train_evaluate(             \
    train_paths,                \
    test_paths,                 \
    training_X_pt_path,         \
    training_y_pt_path,         \
    training_W_pt_path,         \
    load_training_from_pts,     \
    save_training_to_pts,       \
    hand_name,                  \
    model_path,                 \
    train_model,                \
    save_trained_model,         \
    evaluate_model              ):

    # fingers: thumb, index, middle, ring, pinky
    # give: left/right, finger name, feature
    # Load from folder and process
    
    #############################################################################################
    # PREPARE TRAINING DATA
    #############################################################################################
    # If one is not none, all others are not
    training_pt_paths_given = False

    if training_X_pt_path != None and training_y_pt_path != None and training_W_pt_path != None:
        training_pt_paths_given = True

    if load_training_from_pts:
        assert training_pt_paths_given

        # Load from file
        print("Loading training data from pt files..")
        training = torch.load(training_X_pt_path)
        training_gt = torch.load(training_y_pt_path)
        training_weights = torch.load(training_W_pt_path)
        print("Training data loaded.")
    else: # otherwise, load the data from folders
        # Parse
        print("Parsing training data from recorded files..")

        train_Xs, train_ys = [], []
        train_windows, train_IoU, train_keytap_moments = [], [], []
        
        # Read data from different paths
        for p in train_paths:
            one_X_train, one_y_train, one_windows, one_IoU, one_keytap_moments = process_and_get_np(p, "train", hand_name)
            
            train_Xs.append(one_X_train)
            train_ys.append(one_y_train)

            train_windows.append(one_windows)
            train_IoU.append(one_IoU)
            train_keytap_moments.append(one_keytap_moments)
            
        # concat data - to be used while training 
        X = np.concatenate(train_Xs)
        y = np.concatenate(train_ys)

        # compute freq of samples from each class, compute weights
        count_total = y.shape[0]
        count_ones = np.count_nonzero(y)
        count_zeros = count_total - count_ones
        
        ones_freq = count_ones / count_total
        zeros_freq = count_zeros / count_total

        ones_weight = 1 / ones_freq
        zeros_weight = 1 / zeros_freq

        weights = []
        for i in range(count_total):
            if y[i] == 0:
                weights.append(zeros_weight)
            else:
                weights.append(ones_weight)

        W = np.array(weights)

        # Inform
        print("Number of samples: {} (1: {}, 0: {})".format(count_total, count_ones, count_zeros))

        training = torch.from_numpy(X).float()
        training_gt = torch.from_numpy(y).float().view(-1, 1)
        training_weights = torch.from_numpy(W).float().view(-1, 1)
    
    if save_training_to_pts:
        assert training_pt_paths_given

        print("Saving training data to pts..")
        torch.save(training, training_X_pt_path)
        torch.save(training_gt, training_y_pt_path)
        torch.save(training_weights, training_W_pt_path)
        print("Training data saved.")


    #############################################################################################
    # PREPARE TEST DATA
    #############################################################################################
    test_Xs, test_ys = [], []
    test_windows, test_IoU, test_keytap_moments = [], [], []
    for p in test_paths:
        # test
        one_X_test, one_y_test, one_windows, one_IoU, one_keytap_moments = process_and_get_np(p, "test", hand_name)
        
        test_Xs.append(one_X_test)
        test_ys.append(one_y_test)

        test_windows.append(one_windows)
        test_IoU.append(one_IoU)
        test_keytap_moments.append(one_keytap_moments)
    
    # Concat text data. Be careful that keytap moments and windows need special handling

    # Make it safe to concat test data by making windows and keytap moments continuous
    to_add = 0 # the data will be considered as a whole - therefore, keytap moments and windows should be continuous since they will act as indices
    for one_test_windows, one_test_keytap_moments in zip(test_windows, test_keytap_moments):
        if to_add != 0:
            # add to windows
            for i in range(len(one_test_windows)):
                one_test_windows[i] = one_test_windows[i][0] + to_add, one_test_windows[i][1] + to_add
            
            # add to keytap moments
            for i in range(len(one_test_keytap_moments)):
                one_test_keytap_moments[i] += to_add

        to_add = one_test_windows[-1][1] + 100 # make them distinct so that one IoU will be found for different records
    
    # concat data - to be used while training 
    test_X = np.concatenate(test_Xs)
    test_y = np.concatenate(test_ys)
    test_windows = reduce(lambda x,y: x+y, test_windows)
    test_IoU = reduce(lambda x,y: x+y, test_IoU)
    test_keytap_moments = reduce(lambda x,y: x+y, test_keytap_moments)
    
    # Save npy
    # np.save("inputs_1.npy", X)
    # np.save("labels_1.npy", y)

    # Load npy
    # X = np.load("inputs_1.npy")
    # y = np.load("labels_1.npy")

    # Create model
    net = Net().cuda()

    # Load model if exists
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    # TODO: Better splitting mechanism is needed, maybe at folder level..
    count_total = training.shape[0]
    count_training = int(count_total * 0.8)
    count_validation = count_total - count_training

    validation = (training[:count_validation, ...]).cuda()
    validation_gt = (training_gt[:count_validation, ...]).cuda()
    validation_weights = (training_weights[:count_validation, ...]).cuda()

    training = training[count_validation:, ...]
    training_gt = training_gt[count_validation:, ...]
    training_weights = training_weights[count_validation:, ...]

    ##########################################################################################
    # TRAINING 
    ##########################################################################################
    if train_model:
        num_epochs = 10000
        regularization_strength = 1e-1
        optimizer = torch.optim.Adam(net.parameters(), weight_decay=regularization_strength, lr=1e-4)

        train_loss_history = []
        valid_loss_history = []

        # TODO: Set it wisely
        batch_size = 100000

        try:
            for epoch in range( 0, num_epochs ):

                # Use data loader to create mini-batches
                train_X_loader = DataLoader(training, batch_size=batch_size)
                train_y_loader = DataLoader(training_gt, batch_size=batch_size)
                train_W_loader = DataLoader(training_weights, batch_size=batch_size)

                # accumulate the loss for each epoch
                train_loss_acc = 0

                for train_X_ram, train_y_ram, train_W_ram in zip(train_X_loader, train_y_loader, train_W_loader):
                    # move the mini batch to GPU
                    train_X_gpu = train_X_ram.cuda()
                    train_y_gpu = train_y_ram.cuda()
                    train_W_gpu = train_W_ram.cuda()

                    hypo = net(train_X_gpu)

                    # create criterion using weights
                    criterion =  nn.BCELoss(weight=train_W_gpu) # TODO: Use BCEWithLogitsLoss later

                    # compute loss
                    # here is the problem
                    loss = criterion(hypo, train_y_gpu)

                    train_loss_acc += loss.item() * (train_X_gpu.shape[0] / training.shape[0] )

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                    if save_trained_model and epoch % 100 == 0:
                        # modify the model path and populate the epoch id
                        checkpoint_path = ''.join(list(model_path)[:])
                        checkpoint_path = checkpoint_path.replace('.pt', '-{}.pt'.format(epoch))
                        torch.save(net.state_dict(), checkpoint_path)

                    # free memory on GPU
                    del train_X_gpu, train_y_gpu, train_W_gpu

                # End of an epoch

                # Save training loss
                train_loss_history.append(train_loss_acc)

                # Validation
                hypo = net(validation)
                
                # create criterion using weights
                criterion =  nn.BCELoss(weight=validation_weights) # TODO: Use BCEWithLogitsLoss later

                # compute loss
                loss = criterion(hypo, validation_gt)

                valid_loss = loss.item()

                valid_loss_history.append(valid_loss)

                # Report
                if epoch % 10 == 0:
                    print("epoch {:5d}, training_loss: {:.10f}, validation_loss: {:.10f}".format(epoch, train_loss_acc, valid_loss), flush=True )

                train_loss_acc = 0


            # Finally, save the model again
            if save_trained_model:
                torch.save(net.state_dict(), model_path)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Saving the model if necessary and terminating..")
            # Finally, save the model again
            if save_trained_model:
                torch.save(net.state_dict(), model_path)
            exit()

    if not evaluate_model:
        return
    
    # Evaluate the model on different thresholds and obtain the evaluation results - to be interpreted later
    results = evaluate(net, test_X, test_y, test_windows, test_keytap_moments)

    # Interpret the evaluations results - plots etc.
    interpret_evaluation_results(results)

def are_windows_intersecting(w1, w2):
    return not ( w1[0] >= w2[1] or w2[0] >= w1[1] )
    
# to be thresholded later if need be
def nms(hypo, windows):
    #hypo = list(enumerate(hypo))
    #windows = list(enumerate(windows))

    packed = list(enumerate(zip(hypo, windows))) # (index, (hypo, (wstart, wend)))
    #print("first of packed:", packed[0])
    #print("key for confidence ordering: ",  packed[0][1][1])

    #packed_window_ordered = sorted(packed, key=lambda x: x[1][1]) # TODO: To be used while binary serach in below code
    packed_confidence_ordered = sorted(packed, key=lambda x: x[1][0], reverse=True)

    #print("first of ordered:", packed_confidence_ordered[0])
    
    # processed means either suppressed or added to the resulting list due to being the max around
    processed = set() 

    # the incides of those that are not suppressed
    max_index = set()

    # from the one with the highest prob to the lowest
    for i, (confidence, window) in packed_confidence_ordered:
        if i in processed:
            continue
        
        # newly processing
        processed.add(i)

        # add to the result - this is the max around since it is not processed earlier
        max_index.add(i)

        # remove the intersecting results as this one has the highest confidence
        # removal is done implicity by adding those windows to processed set
        # TODO: binary search could be done here
        for i2, (_, window2) in packed_confidence_ordered:
            # if any other data intersects with the max found, make it processed to make sure 
            # it is suppressed (not added to the result later)
            if are_windows_intersecting(window, window2):
                processed.add(i2)
    
    result = []

    for i, h in enumerate(hypo):
        if i in max_index:
            result.append(h)
        else:
            result.append(0)

    # put in index order
    return result

# given the windows, the keytap moments, and the IoU threshold for a window being labeled as positive, get the
# labels for all windows
def get_groundtruth_labels(windows, keytap_moments, true_positive_IoU_threshold):
    window_len = windows[0][1] - windows[0][0]

    IoU = get_IoU_for_windows(windows, keytap_moments, window_len, is_window=True)

    # label by considering true_positive_IoU_threshold
    # list of tuples, each tuple as (IoU, corresponding gt index) where IoU is < 1 and > 0
    for i in range(len(IoU)):
        IoU[i] = 1 if IoU[i] >= true_positive_IoU_threshold else 0
    labels = IoU
    
    return labels

# hypo: 0s and 1s, len: windows
# midpoints: the midpoints of windows
# keytap_moments: the midpoints of keytap moments

# associate positive predictions with the ground truths, such that, we would have 
# a list of elements (hypo index, ground truth index, IoU)
def associate_positive_preds_with_groundtruthts(hypo, midpoints, keytap_moments, window_len, true_positive_IoU_threshold):
    associations = []
    
    for hypo_ind, hm in enumerate(zip(hypo, midpoints)): # each hypo is associated with a midpoint
        h, m = hm
        # if the hypo is a positive, we will need an association for that
        if h == 1:
            # search the key tap moment that is closest to the hypo
            distances = list(enumerate( [ abs(m - keytap_moment) for keytap_moment in keytap_moments ] ))

            # find the closest
            min_dist = min(distances, key=lambda x: x[1])
            keytap_moment_index, min_dist = min_dist

            # compute the IoU
            intersection = 0 if min_dist > (window_len / 2) else ((window_len / 2) - min_dist) / (window_len / 2)

            # check if it meets the threshold
            if intersection >= true_positive_IoU_threshold:
                # add to the list of associations
                association = ( hypo_ind, keytap_moment_index, intersection )
                associations.append( association )

    # go over all associations. if there is more than one association for a groundtruth, keep the one with the maximum IoU
    associations.sort(key=lambda x: x[2]) # sort about IoU

    association_to_remove_hypo_indices = set()
    associated_gt_index = set()

    for association in associations:
        if association[1] not in associated_gt_index:
            # the association for gt that has the highest IoU for that gt
            associated_gt_index.add(association[1])
            continue
        else:
            # to be removed from the list
            association_to_remove_hypo_indices.add(association[0])

    old_associations = associations
    associations = []

    for assocation in old_associations:
        if assocation[0] not in association_to_remove_hypo_indices:
            associations.append(association)
    
    return associations

# use the network, then do nms, then use classification threshold
# return the resulting hypo labels: 1 if key tap, 0 if not
def predict_keytap_windows(net, X, windows, classification_threshold=0.5):
    window_len = abs(windows[0][0] - windows[0][1])
    window_mid_point = [ (w[0] + w[1]) // 2  for w in windows]

    hypo = net(torch.from_numpy(X).float().cuda()).detach().cpu().numpy().tolist()
    hypo = list(map( lambda x: x[0], hypo ))

    nm_suppressed = nms(hypo, windows)

    hypo_labels = [ 1 if h >= classification_threshold else 0 for h in nm_suppressed ]

    return hypo_labels

def interpret_evaluation_results(results):
    # results[IoU_threshold]: result for a IoU threshold
    # Fill fpr, tpr, thresholds for roc curve

    #fpr, tpr, thresholds, auc = {}, {}, {}, {}
    recall, precision, thresholds, auc = {}, {}, {}, {}

    # prepare list of fpr, tpr, and thresholds for each IoU
    for IoU_threshold in results:
        recall[IoU_threshold] = []
        precision[IoU_threshold] = []
        thresholds[IoU_threshold] = []

        for result in results[IoU_threshold]:
            #print("Threshold:", result["classification_threshold"], ", tpr:", result["tpr"], ", fpr:",result["fpr"] )
            recall[IoU_threshold].append(result["recall"])
            precision[IoU_threshold].append(result["precision"])
            thresholds[IoU_threshold].append(result["classification_threshold"])
        
        # Compute AUC   
        auc[IoU_threshold] = metrics.auc(recall[IoU_threshold], precision[IoU_threshold])


    # Compute AUC
    #auc = metrics.auc(fpr, tpr)
    #print("AUC is:", auc)

    # Draw ROC
    plt.figure()
    lw = 2

    some_number = random.randint(0, 100)
    print('The number is: ', some_number)
    for IoU_threshold in auc:
        
        # write to file for plotting in GNUPlot
        string_to_write = ''
        for r, p in zip(recall[IoU_threshold], precision[IoU_threshold]):
            string_to_write += '{} {}\n'.format(r, p)
        
        file_p = '{}-{}.dat'.format(some_number, IoU_threshold)
        with open(file_p, 'w+') as f:
            f.write(string_to_write)
        # End of writing to file


        plt.plot(recall[IoU_threshold], precision[IoU_threshold], lw=lw, label='IoU threshold = {:.2f} (AUC = {:.2f})'.format(IoU_threshold, auc[IoU_threshold]))
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    plt.xlabel('Recall')
    #plt.ylabel('True Positive Rate')
    plt.ylabel('Precision')
    #plt.title('Receiver operating characteristics')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# hypo: list of ones and zeros, one corresponds to each windows. after nms.
# positions: 
def evaluate_with_given_hypo(hypo, positions, windows, keytap_moments, true_positive_IoU_thresholds=[ 0.5, 0.75, 1], classification_thresholds=[ 0.5, 0.75, 1]):
    window_len = abs(windows[0][0] - windows[0][1])

    window_mid_point = [ (w[0] + w[1]) // 2  for w in windows]

    hypo = list(map( lambda x: x[0], hypo ))

    # do non-max supression: set the result around maximum values to 0 (suppress them)
    nm_suppressed = nms(hypo, windows)

    # Each element is a dictionary with keys: "classification_threshold", "tp", "fp", "tn", "fn"
    results = { }
    for IoU_threshold in true_positive_IoU_thresholds:
        results[IoU_threshold] = []

    # Get results for each unique classification prop
    # if the hypo makes a result leq 'classification_threshold', it is told to be 0, otherwise 1
    for classification_threshold in classification_thresholds:
        # result = { "classification_threshold" : classification_threshold }

        # # apply classification threshold to get resulting hypo labels: 0s and 1s
        # hypo_labels = [ 1 if h >= classification_threshold else 0 for h in nm_suppressed ]
        
        for IoU_threshold in true_positive_IoU_thresholds:
            result = { "classification_threshold" : classification_threshold }

            # apply classification threshold to get resulting hypo labels: 0s and 1s
            hypo_labels = [ 1 if h >= classification_threshold else 0 for h in nm_suppressed ]

            # for each prediction having value 1 (positive, key tap found), associate it with the closest groundtruth
            # and find their intersection over union. if it cannot be associated with a groundtruth (IoU is 0), then, set
            # the associated to groundtruth index to 0
            # True positives
            # a list of elements (hypo index(hypo labels or windows), ground truth index (keytap_moments), IoU (btw 0 and 1))
            associations = associate_positive_preds_with_groundtruthts(hypo_labels, window_mid_point, keytap_moments, window_len, IoU_threshold)
            
            # True Positive : Hypo positive that is successfully associated to a ground truth positive: associations list
            # False Positive: Hypo labels that couldn't be associated to a ground truth positive: hypo labels that are not associated
            # False Negative: Ground truth labels that couldn't be associated to any hypo labels
            # True Negative: Negative count from hypo labels - false negative

            # TODO: Create below values in advance, calculate them for each batch and accumulate, then, do evaluation on overall values
            # Find the false positives
            predicted_positives_count = sum(hypo_labels)
            predicted_negatives_count = len(hypo_labels) - predicted_positives_count

            gt_positives_count = len(keytap_moments)
            gt_negatives_count = len(window_mid_point) - gt_positives_count

            true_positive_count = len(associations)
            false_positive_count = predicted_positives_count - true_positive_count

            false_negative_count = gt_positives_count - true_positive_count
            true_negative_count = predicted_negatives_count - false_negative_count

            # for using sklearn, create y_pred and y_score that would be compatible with the results we found
            y_true = []
            y_pred = []

            # true positives
            y_true.extend( [1] * true_positive_count )
            y_pred.extend( [1] * true_positive_count )

            # false positives
            y_true.extend( [0] * false_positive_count )
            y_pred.extend( [1] * false_positive_count )
            
            # true negatives
            y_true.extend( [0] * true_negative_count )
            y_pred.extend( [0] * true_negative_count )

            # false negatives
            y_true.extend( [1] * false_negative_count )
            y_pred.extend( [0] * false_negative_count )

            confusion_matrix = metrics.confusion_matrix( y_true, y_pred, [0, 1])

            tn, fp, fn, tp = confusion_matrix.ravel()

            # print("predicted_positives_count", predicted_positives_count)
            # print("predicted_negatives_count", predicted_negatives_count)
            # print("gt_positives_count", gt_positives_count)
            # print("gt_negatives_count", gt_negatives_count)
            # print("true_positive_count", true_positive_count, tp)
            # print("false_positive_count", false_positive_count, fp)
            # print("false_negative_count", false_negative_count, fn)
            # print("true_negative_count", true_negative_count, tn)

            # Fill the result
            result["tp"] = true_positive_count
            result["fp"] = false_positive_count
            result["tn"] = true_negative_count
            result["fn"] = false_negative_count
            result["tpr"] = true_positive_count / (true_positive_count + false_negative_count)
            result["tnr"] = true_negative_count / (true_negative_count + false_positive_count)
            result["fpr"] = false_positive_count / (false_positive_count + true_negative_count)
            result["fnr"] = false_negative_count / (false_negative_count + true_positive_count)
            
            if (true_positive_count + false_positive_count) != 0:
                result["precision"] = true_positive_count / (true_positive_count + false_positive_count)
            else:
                result["precision"] = 1

            if (true_positive_count + false_negative_count) != 0:
                result["recall"] = true_positive_count / (true_positive_count + false_negative_count)
            else:
                result["recall"] = 1

            # Find the explicit points
            true_positives_hypo_indices = set()
            true_positives_gt_indices = set()
            
            for hypo_index, gt_index, IoU in associations:
                true_positives_hypo_indices.add(hypo_index)
                true_positives_gt_indices.add(gt_index)
            
            # Group true positives and false negatives in gt
            true_positives_gt_points = []
            false_negatives_gt_points = []
            for i, keytap_moment in enumerate(keytap_moments):
                if i in true_positives_gt_indices:
                    true_positives_gt_points.append(keytap_moment)
                else:
                    false_negatives_gt_points.append(keytap_moment)
            
            # Group true positives and false positives in hypo
            true_positives_hypo_points = []
            false_positives_hypo_points = []
            for i, (midpoint, hypo_label) in enumerate(zip(window_mid_point, hypo_labels)):
                if hypo_label == 1:
                    if i in true_positives_hypo_indices:
                        true_positives_hypo_points.append(midpoint)
                    else:
                        false_positives_hypo_points.append(midpoint)

            results[IoU_threshold].append(result)

    return results

# the line is from p0 to p1
# Returns None if no intersection found
# Returns vec if there is an intersection
def check_vector(pn, pp0, l0, l1):
    vec_btw = [l1x - l0x for l1x, l0x in zip(l1, l0)]
    l = vec_btw
    d = vec_len(l)

    if d == 0:
        return None

    l = get_normalized_vec(l)

    # line eq: p = l0 + l d
    
    dprime_top = dot_product(vec_diff(pp0,l0), pn)
    dprime_bottom = dot_product(l, pn)

    dprime = dprime_top / dprime_bottom
    # / 10 cm 
    intersects = dprime <= d and dprime >= 0
    #intersects = dprime <= (d+5) and dprime >= -5

    if not intersects:
        return None
    else:
        intersection_point = vec_sum(l0, [x*dprime for x in l])
        return intersection_point

# returns indices and the corresponding positions
def check_whole_recording(plane_normal, plane_point, whole_positions):
    intersecting_indices = []
    intersecting_positions = []
    for i in range( len(whole_positions) - 1 ):
        pos = check_vector(plane_normal, plane_point, whole_positions[i], whole_positions[i+1] )

        if pos != None:
            intersecting_indices.append(i)
            intersecting_positions.append(pos)
    
    return intersecting_indices, intersecting_positions

def get_windows_from_intersecting_points(intersecting_indices, intersecting_positions, window_len):
    windows = []
    windows_weights = []
    corresponding_intersecting_positions = []

    window_len //= 2

    window_mid = window_len

    i = 0
    
    while True:
        ii = i

        window_weight = 0

        window_start = window_mid - window_len
        window_end = window_mid + window_len

        # Check all close intersecting indices (ii)
        while True:
            intersecting_moment = intersecting_indices[ii]

            if intersecting_moment > (window_mid - window_len) and intersecting_moment < (window_mid + window_len):
                # included in the window
                window_weight += 1 * 1000

                # increase weight with the square of the min dist
                window_weight += min( pow(intersecting_moment - window_start, 2), pow(intersecting_moment - window_end, 2)  )

                ii += 1

                if ii >= len(intersecting_indices):
                    break
            else:
                break

        if window_weight > 0:
            windows.append((window_start, window_end))
            windows_weights.append(window_weight)
            corresponding_intersecting_positions.append( intersecting_positions[ii - 1] )

        # Check if we need to increase i
        if intersecting_indices[i] <= window_start:
            i += 1

            # Check if we are done with all intersecting indices
            if i >= len(intersecting_indices):
                break
        
        # Continue with the next window
        window_mid += 1

    #print("\nwindows:", windows[:50])
    #print("\nwindows_weights:", windows_weights[:50])
    return windows, windows_weights, corresponding_intersecting_positions


# If intersects, return point. If not, return None
def check_window(plane_normal, plane_point, window, whole_positions):
    window_point_indices = range(window[0], window[1])
    window_points = [ whole_positions[i] for i in window_point_indices ]

    # TODO: Can be optimized
    intersection_points = []
    for i in range(0, len(window_points) -1 ):
        intersection_points.append( check_vector(plane_normal, plane_point, window_points[i], window_points[i+1]) )

    for p in intersection_points:
        if p != None:
            return p

    return None

# if we have 100 instances, whole positions have the finger tip position for each
# if we have 10 key tap moments detected by our model, positive_windows have 10 windows regarding those
# we will create set of vectors for each positive_windows using whole_positions, then, check if they intersect with the plane
# If one does not intersect, report it as false_positive, so that, it can be removed from the positives list 
def check_windows(plane_normal, plane_point, positive_windows, whole_positions):
    return [ check_window(plane_normal, plane_point, window, whole_positions) for window in positive_windows ]

def evaluate_new(hypo_labels, windows, keytap_moments, true_positive_IoU_thresholds=[ 0.25 ]):
    window_len = abs(windows[0][0] - windows[0][1])
    window_mid_point = [ (w[0] + w[1]) // 2  for w in windows]

    # Each element is a dictionary with keys: "classification_threshold", "tp", "fp", "tn", "fn"
    results = []


    for IoU_threshold in true_positive_IoU_thresholds:
        result = { 'IoU_threshold' : IoU_threshold }
        associations = associate_positive_preds_with_groundtruthts(hypo_labels, window_mid_point, keytap_moments, window_len, IoU_threshold)
        
        predicted_positives_count = sum(hypo_labels)
        predicted_negatives_count = len(hypo_labels) - predicted_positives_count

        gt_positives_count = len(keytap_moments)
        gt_negatives_count = len(window_mid_point) - gt_positives_count

        true_positive_count = len(associations)
        false_positive_count = predicted_positives_count - true_positive_count

        false_negative_count = gt_positives_count - true_positive_count
        true_negative_count = predicted_negatives_count - false_negative_count

        # for using sklearn, create y_pred and y_score that would be compatible with the results we found
        y_true = []
        y_pred = []

        # true positives
        y_true.extend( [1] * true_positive_count )
        y_pred.extend( [1] * true_positive_count )

        # false positives
        y_true.extend( [0] * false_positive_count )
        y_pred.extend( [1] * false_positive_count )
            
        # true negatives
        y_true.extend( [0] * true_negative_count )
        y_pred.extend( [0] * true_negative_count )

        # false negatives
        y_true.extend( [1] * false_negative_count )
        y_pred.extend( [0] * false_negative_count )

        confusion_matrix = metrics.confusion_matrix( y_true, y_pred, [0, 1])

        tn, fp, fn, tp = confusion_matrix.ravel()

        # Fill the result
        result["tp"] = true_positive_count
        result["fp"] = false_positive_count
        result["tn"] = true_negative_count
        result["fn"] = false_negative_count
        result["tpr"] = true_positive_count / (true_positive_count + false_negative_count)
        result["tnr"] = true_negative_count / (true_negative_count + false_positive_count)
        result["fpr"] = false_positive_count / (false_positive_count + true_negative_count)
        result["fnr"] = false_negative_count / (false_negative_count + true_positive_count)
        result["acc"] = (true_positive_count + true_negative_count) / (true_positive_count + true_negative_count + false_positive_count + false_negative_count)

        if (true_positive_count + false_positive_count) != 0:
            result["precision"] = true_positive_count / (true_positive_count + false_positive_count)
        else:
            result["precision"] = 1

        if (true_positive_count + false_negative_count) != 0:
            result["recall"] = true_positive_count / (true_positive_count + false_negative_count)
        else:
            result["recall"] = 1

        # Find the explicit points
        true_positives_hypo_indices = set()
        true_positives_gt_indices = set()
            
        for hypo_index, gt_index, IoU in associations:
            true_positives_hypo_indices.add(hypo_index)
            true_positives_gt_indices.add(gt_index)
            
        # Group true positives and false negatives in gt
        true_positives_gt_points = []
        false_negatives_gt_points = []
        for i, keytap_moment in enumerate(keytap_moments):
            if i in true_positives_gt_indices:
                true_positives_gt_points.append(keytap_moment)
            else:
                false_negatives_gt_points.append(keytap_moment)
            
        # Group true positives and false positives in hypo
        true_positives_hypo_points = []
        false_positives_hypo_points = []
        for i, (midpoint, hypo_label) in enumerate(zip(window_mid_point, hypo_labels)):
            if hypo_label == 1:
                if i in true_positives_hypo_indices:
                    true_positives_hypo_points.append(midpoint)
                else:
                    false_positives_hypo_points.append(midpoint)

        results.append(result)

    return results


    


    

# if the hypo makes an IoU of 'true_positive_IoU_threshold' with a groundtruth moment, it is counted as a true positive
# keytap_moments: time instances, not indices
# If defined, evaluate only for the given list of classification thresholds
def evaluate(net, X, y, windows, keytap_moments, true_positive_IoU_thresholds=[ 0.5, 0.75, 1 ]):
    # Compute the window length
    window_len = abs(windows[0][0] - windows[0][1])
    
    # Compute the window midpoints. For example, for a window of (0, 50), make it 25
    window_mid_point = [ (w[0] + w[1]) // 2  for w in windows]

    # Go into the evaluation mode (so that, the network will work efficiently)
    net.eval()

    # Do prediction on input (X) using the network (net)
    print("Doing prediction for test.")
    with torch.no_grad():
        hypo = net(torch.from_numpy(X).float().cuda()).cpu().numpy().tolist()
    print("Prediction for test was done.")
    # hypo includes confidence scores for each window
    hypo = list(map( lambda x: x[0], hypo ))

    # Do non-max supression: set the result around maximum values to 0 (suppress them)
    nm_suppressed = nms(hypo, windows)

    # Get unique classification thresholds(proposals) for ROC and AUC
    unique_classification_props = set(nm_suppressed)
    
    # Add the minimum and the maximum values
    unique_classification_props.add(0 - sys.float_info.epsilon)
    unique_classification_props.add(1 + sys.float_info.epsilon)

    unique_classification_props = sorted(list(unique_classification_props))

    unique_classification_props.sort()

    # Each element is a dictionary with keys: "classification_threshold", "tp", "fp", "tn", "fn"
    results = { }
    for IoU_threshold in true_positive_IoU_thresholds:
        results[IoU_threshold] = []
    
    # Get results for each unique classification prop
    # if the hypo makes a result leq 'classification_threshold', it is told to be 0, otherwise 1
    for classification_threshold in unique_classification_props:
        for IoU_threshold in true_positive_IoU_thresholds:
            result = { "classification_threshold" : classification_threshold }

            # Apply classification threshold to get resulting hypo labels: 0s and 1s
            hypo_labels = [ 1 if h >= classification_threshold else 0 for h in nm_suppressed ]

            # for each prediction having value 1 (positive, key tap found), associate it with the closest groundtruth
            # and find their intersection over union. if it cannot be associated with a groundtruth (IoU is 0), then, set
            # the associated to groundtruth index to 0
            # True positives
            # a list of elements (hypo index(hypo labels or windows), ground truth index (keytap_moments), IoU (btw 0 and 1))
            associations = associate_positive_preds_with_groundtruthts(hypo_labels, window_mid_point, keytap_moments, window_len, IoU_threshold)
            
            # True Positive : Hypo positive that is successfully associated to a ground truth positive: associations list
            # False Positive: Hypo labels that couldn't be associated to a ground truth positive: hypo labels that are not associated
            # False Negative: Ground truth labels that couldn't be associated to any hypo labels
            # True Negative: Negative count from hypo labels - false negative

            # TODO: Create below values in advance, calculate them for each batch and accumulate, then, do evaluation on overall values
            # Find the false positives
            predicted_positives_count = sum(hypo_labels)
            predicted_negatives_count = len(hypo_labels) - predicted_positives_count

            gt_positives_count = len(keytap_moments)
            gt_negatives_count = len(window_mid_point) - gt_positives_count

            true_positive_count = len(associations)
            false_positive_count = predicted_positives_count - true_positive_count

            false_negative_count = gt_positives_count - true_positive_count
            true_negative_count = predicted_negatives_count - false_negative_count

            # for using sklearn, create y_pred and y_score that would be compatible with the results we found
            y_true = []
            y_pred = []

            # true positives
            y_true.extend( [1] * true_positive_count )
            y_pred.extend( [1] * true_positive_count )

            # false positives
            y_true.extend( [0] * false_positive_count )
            y_pred.extend( [1] * false_positive_count )
            
            # true negatives
            y_true.extend( [0] * true_negative_count )
            y_pred.extend( [0] * true_negative_count )

            # false negatives
            y_true.extend( [1] * false_negative_count )
            y_pred.extend( [0] * false_negative_count )

            confusion_matrix = metrics.confusion_matrix( y_true, y_pred, [0, 1])

            tn, fp, fn, tp = confusion_matrix.ravel()

            # print("predicted_positives_count", predicted_positives_count)
            # print("predicted_negatives_count", predicted_negatives_count)
            # print("gt_positives_count", gt_positives_count)
            # print("gt_negatives_count", gt_negatives_count)
            # print("true_positive_count", true_positive_count, tp)
            # print("false_positive_count", false_positive_count, fp)
            # print("false_negative_count", false_negative_count, fn)
            # print("true_negative_count", true_negative_count, tn)

            # Fill the result
            result["tp"] = true_positive_count
            result["fp"] = false_positive_count
            result["tn"] = true_negative_count
            result["fn"] = false_negative_count
            result["tpr"] = true_positive_count / (true_positive_count + false_negative_count)
            result["tnr"] = true_negative_count / (true_negative_count + false_positive_count)
            result["fpr"] = false_positive_count / (false_positive_count + true_negative_count)
            result["fnr"] = false_negative_count / (false_negative_count + true_positive_count)
            
            if (true_positive_count + false_positive_count) != 0:
                result["precision"] = true_positive_count / (true_positive_count + false_positive_count)
            else:
                result["precision"] = 1

            if (true_positive_count + false_negative_count) != 0:
                result["recall"] = true_positive_count / (true_positive_count + false_negative_count)
            else:
                result["recall"] = 1

            # Find the explicit points
            true_positives_hypo_indices = set()
            true_positives_gt_indices = set()
            
            for hypo_index, gt_index, IoU in associations:
                true_positives_hypo_indices.add(hypo_index)
                true_positives_gt_indices.add(gt_index)
            
            # Group true positives and false negatives in gt
            true_positives_gt_points = []
            false_negatives_gt_points = []
            for i, keytap_moment in enumerate(keytap_moments):
                if i in true_positives_gt_indices:
                    true_positives_gt_points.append(keytap_moment)
                else:
                    false_negatives_gt_points.append(keytap_moment)
            
            # Group true positives and false positives in hypo
            true_positives_hypo_points = []
            false_positives_hypo_points = []
            for i, (midpoint, hypo_label) in enumerate(zip(window_mid_point, hypo_labels)):
                if hypo_label == 1:
                    if i in true_positives_hypo_indices:
                        true_positives_hypo_points.append(midpoint)
                    else:
                        false_positives_hypo_points.append(midpoint)

            results[IoU_threshold].append(result)

    return results

# Returns an array of indices each of which corresponds to moments of key taps
def keytap_inference(hand_name, recording_folder, model_path, classification_threshold):
    # Load model
    assert os.path.exists(model_path)

    net = Net().cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # TODO X, windows (the input windows, each (start, end))
    # Load data
    p = recording_folder
    X, dummy_y, windows, dummy_IoU, dummy_keytap_moments = process_and_get_np(p, "test", hand_name)
    
    with torch.no_grad():
        hypo = net(torch.from_numpy(X).float().cuda()).cpu().numpy().tolist()

    hypo = list(map( lambda x: x[0], hypo ))

    nm_suppressed = nms(hypo, windows)
    hypo_labels = [ 1 if h >= classification_threshold else 0 for h in nm_suppressed ]

    weights = []
    for h, n in zip(hypo_labels, nm_suppressed):
        if h == 1:
            weights.append(n)

    # hypo labels, windows, keytap_moments, window_len, IoU_threshold
    indices = []
    positive_windows = []
    for w, h in zip(windows, hypo_labels):
        if h == 1:
            indices.append( (w[0]+w[1]) // 2 )
            positive_windows.append(w)

    # TODO Remove
    # indices = dummy_keytap_moments

    # get the tip positions from another file and pull the key tap positions
    finger_names = ["index"]
    # feature_names = ["tip_pos_x", "tip_pos_y", "tip_pos_z"]
    feature_names = ["lm_pos_x", "lm_pos_y", "lm_pos_z"]
    
    index_tip_pos_stacked = []
    for finger in finger_names:
        for feature in feature_names:
            filepath = get_full_filepath(recording_folder, hand_name, finger, feature)
            data_string_list = read_data_from_tab_delimited_file(filepath)
            data_float_list = list(map(float, data_string_list))

            index_tip_pos_stacked.append(data_float_list)
    
    index_tip_pos_stacked_np = np.array(index_tip_pos_stacked)
    index_tip_pos_stacked_np = np.swapaxes(index_tip_pos_stacked,0,1)
    positions = index_tip_pos_stacked_np[indices, :]


    if len(dummy_keytap_moments) > 0:
        print("\nBEFORE REFINING {}:".format(hand_name.upper()))
        results = evaluate_new(hypo_labels, windows, dummy_keytap_moments)
        pprint.pprint(results)

        whole_results[hand_name]['before_refining'][rec_fold] = results
        print(" ")
    else:
        print("No ground truth data can be found for evaluation.")

    # TODO: Changes start here
    points = positions

    # Plane fitting
    points_for_plane_fitting = points

    
    # Weights for plane fitting
    weights = [ pow(2, w * 10) for w in weights]
    max_w = 1 if not weights else max(weights)
    weights = [ w / max_w for w in weights ]

    # coeffs = [ int(pow(2, w * 10)) for w in weights ]
    # points_for_plane_fitting = []
    # for coeff, point in zip(coeffs, points.tolist()):
    #     points_for_plane_fitting.extend( [point] * coeff )
    #plane_normal, plane_point = fit_plane( np.array( points_for_plane_fitting) )

    return positive_windows, index_tip_pos_stacked_np, indices, windows, dummy_keytap_moments, points_for_plane_fitting, weights
    
    # Now that we found the plane, we can check all the points against the plane
    # intersecting_indices, intersecting_positions = check_whole_recording(plane_normal, plane_point, index_tip_pos_stacked_np.tolist())
    # Create windows with key taps in it
    # new_positive_windows, window_weights, corresponding_intersecting_positions = get_windows_from_intersecting_points(intersecting_indices, intersecting_positions, 30) # TODO: Window len
    # Apply non-max suppression -- eliminate ones including the same key taps etc.
    # suppressed_windows_weights = nms(window_weights, new_positive_windows)
    # suppressed_windows = []
    # for i, w in enumerate(new_positive_windows):
    #     if suppressed_windows_weights[i] != 0:
    #         suppressed_windows.append(w)
    # Get corresponding positions as well
    # set_supressed_windows = set(suppressed_windows)
    # positions_for_supressed_windows = []
    # for w, point in zip(new_positive_windows, corresponding_intersecting_positions):
    #     if w in set_supressed_windows:
    #         positions_for_supressed_windows.append(point)
    # Finally we have the positive windows and the corresponding positions
    # positive_windows = suppressed_windows
    # positions = positions_for_supressed_windows
    # indices = [ (w[0] + w[1]) // 2 for w in positive_windows ]
    
    
    # total_count = index_tip_pos_stacked_np.shape[0]
    # whole_windows_with_step_30 = []
    # whole_indices = []

    # window_size = 30
    # step_size = 10
    # for i in range(0, total_count - window_size - 1, step_size):
    #     whole_windows_with_step_30.append((i, i+window_size))

    #     whole_indices.append( i + (window_size // 2)  )

# positive_windows index_tip_pos_stacked_np indices 
def continue_with_plane(plane_normal, plane_point, positive_windows, index_tip_pos_stacked_np, indices, windows, gt_keytap_moments, hand_name, inlier_mask):
    windows_intersection_points = check_windows(plane_normal, plane_point, positive_windows, index_tip_pos_stacked_np.tolist())

    #inlier_mask.tolist()

    new_indices = []
    new_points = []
    new_positive_windows = []
    for index, intersection_point, positive_window in zip(indices, windows_intersection_points, positive_windows):
        if intersection_point != None:
            new_indices.append(index)
            new_points.append(intersection_point)
            new_positive_windows.append(positive_window)

    # new_indices = []
    # new_points = []
    # new_positive_windows = []
    # for index, is_inlier, positive_window in zip(indices, inlier_mask, positive_windows):
    #     if is_inlier:
    #         new_indices.append(index)
    #         new_points.append( index_tip_pos_stacked_np[ (positive_window[0] + positive_window[1]) // 2 ] )
    #         new_positive_windows.append(positive_window)


    set_new_positive_windows = set(new_positive_windows)
    new_hypo_labels = [ 1 if window in set_new_positive_windows else 0 for window in windows ]
    

    positions = np.array(new_points)
    indices = new_indices

    if len(gt_keytap_moments) > 0:
        print("\nAFTER REFINING {}:".format(hand_name.upper()))
        results = evaluate_new(new_hypo_labels, windows, gt_keytap_moments)
        pprint.pprint(results)
        print(" ")
        whole_results[hand_name]['after_refining'][rec_fold] = results
    else:
        print("No ground truth data can be found for evaluation.")
    # 
    # TODO: Changes end here

    # from labels, get the 3D positions for each key tap
    # index, fingertip position at moment 15
    # (sample count, xyz, fingers, segment, features) = (hypo indices, all, index finger, 15, 2)
    #positions = X[indices, :, 1, 15, 2]

    #print("\nStatistics:")
    #print(recording_folder, hand_name)
    #print("GT: {}, Found: {}\n".format( len(dummy_keytap_moments), len(indices) ))

    #print("GT key tap moments:", dummy_keytap_moments)
    #print("Found key tap moments:", indices)

    return positions, indices

def main_training(train_left, save_left, test_left, train_right, save_right, test_right):
    left_training_X_filepath = 'training_data/left_training_X.pt'
    left_training_y_filepath = 'training_data/left_training_y.pt'
    left_training_W_filepath = 'training_data/left_training_W.pt'

    right_training_X_filepath = 'training_data/right_training_X.pt'
    right_training_y_filepath = 'training_data/right_training_y.pt'
    right_training_W_filepath = 'training_data/right_training_W.pt'

    left_train_paths = [] #glob.glob("../data/records-012320/left/train/*")
    left_test_paths = [] #glob.glob("../data/records-012320/left/test/*")

    right_train_paths = [] # glob.glob("../data/records-012320/right/train/*")
    right_test_paths = [] #glob.glob("../data/records-012320/right/test/*")

    mixed_train_paths = glob.glob("../data/NecipKeytapTraining/train/*")
    mixed_train_paths += glob.glob("../data/UlkuKeytapTraining/train/*")

    mixed_test_paths = glob.glob("../data/NecipKeytapTraining/test/*")
    mixed_test_paths += glob.glob("../data/UlkuKeytapTraining/test/*")


    # add the mixed paths to the train & test paths
    left_train_paths.extend( mixed_train_paths )
    right_train_paths.extend( mixed_train_paths )

    left_test_paths.extend( mixed_test_paths )
    right_test_paths.extend( mixed_test_paths )

    # RIGHT
    if train_right or test_right:
        print("Running train_evaluate() for right..")
        train_evaluate(                                             \
            right_train_paths,                                      \
            right_test_paths,                                       \
            training_X_pt_path=right_training_X_filepath,          \
            training_y_pt_path=right_training_y_filepath,          \
            training_W_pt_path=right_training_W_filepath,          \
            load_training_from_pts=True,                         \
            save_training_to_pts=False,                         \
            hand_name='right',                                   \
            model_path=right_model_path,                         \
            train_model=train_right,                                  \
            save_trained_model=save_right,                            \
            evaluate_model=test_right                            \
            )
        print("Finished train_evaluate() for right.")

    # LEFT
    if train_left or test_left:
        print("Running train_evaluate() for left..")
        train_evaluate(                                             \
            left_train_paths,                                      \
            left_test_paths,                                       \
            training_X_pt_path=left_training_X_filepath,          \
            training_y_pt_path=left_training_y_filepath,          \
            training_W_pt_path=left_training_W_filepath,          \
            load_training_from_pts=True,                         \
            save_training_to_pts=False,                         \
            hand_name='left',                                   \
            model_path=left_model_path,                         \
            train_model=train_left,                                  \
            save_trained_model=save_left,                            \
            evaluate_model=test_left                            \
            )
        print("Finished train_evaluate() for left.")



def collate_positions(left_positions, left_indices, right_positions, right_indices):
    left_i = 0
    right_i = 0
    left_max = len(left_indices)
    right_max = len(right_indices)

    left_positions = left_positions.tolist()
    right_positions = right_positions.tolist()

    collated_positions = []

    while left_i < left_max and right_i < right_max:
        left_index = left_indices[left_i]
        right_index = right_indices[right_i]

        if left_index < right_index:
            # take the left position as it precedes
            collated_positions.append( left_positions[left_i] )
            left_i += 1
        else:
            collated_positions.append( right_positions[right_i] )
            right_i += 1
    
    # append the remaining from any of them
    while left_i < left_max:
        collated_positions.append( left_positions[left_i] )
        left_i += 1

    while right_i < right_max:
        collated_positions.append( right_positions[right_i] )
        right_i += 1

    return np.array(collated_positions)

def main_inference():
    top_fold = '../data/UserSide/Afsah/'
    sub1s = [ 'Random5', 'Random10', 'Random15', 'Mail']
    sub2s = ['p-{}/'.format(i+1) for i in range(0, 15)]

    paths = []

    for s1 in sub1s:
        for s2 in sub2s:
            p = os.path.join(top_fold, s1, s2)
            paths.append(p)
            assert os.path.exists(p)

    recording_folders = paths

    classification_thresholds = [ 0.5 ]

    for recording_folder in recording_folders:
        print('\n# Doing keytap detection in', recording_folder, flush=True)
        # create inference folder if necessary
        save_folder = os.path.join(recording_folder, 'inference/keytap_detection/')
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        for cls_threshold in classification_thresholds:
            print('\n## With classification threshold:', cls_threshold, flush=True)
            global rec_fold
            rec_fold = recording_folder
            # TODO Open
            positions_file_to_save = 'predicted_keytap_positions_cls{:.2f}_shared_plane_final.npy'.format(cls_threshold)
            indices_file_to_save = 'predicted_keytap_indices_cls{:.2f}_shared_plane_final.txt'.format(cls_threshold)
            # TODO Close
            # positions_file_to_save = 'gt_keytap_positions.npy'
            # indices_file_to_save = 'gt_keytap_indices.txt'
            
            positions_file_to_save = os.path.join(save_folder, positions_file_to_save)
            indices_file_to_save = os.path.join(save_folder, indices_file_to_save)

            left_positive_windows, left_index_tip_pos_stacked_np, left_indices, left_windows, left_gt_keytap_moments, left_points_for_plane_fitting, left_weights = keytap_inference('left', recording_folder, left_model_path, cls_threshold)
            right_positive_windows, right_index_tip_pos_stacked_np, right_indices, right_windows, right_gt_keytap_moments, right_points_for_plane_fitting, right_weights = keytap_inference('right', recording_folder, right_model_path, cls_threshold)
            
            # Record the first positive windows here to be given at each step as is
            # At each step, fit a plane, eliminate points, fit new plane, eliminate points (from scratch)
            # fit plane
            # TODO Open
            plane_normal, plane_point, _ = fit_plane( np.array(left_points_for_plane_fitting.tolist() + right_points_for_plane_fitting.tolist()), np.array( left_weights + right_weights ) )

            # left_plane_normal, left_plane_point, left_inlier_mask = fit_plane( np.array(left_points_for_plane_fitting), np.array(left_weights) )
            # right_plane_normal, right_plane_point, right_inlier_mask = fit_plane( np.array(right_points_for_plane_fitting), np.array(right_weights) )

            # Then continue..
            # TODO Open
            left_positions, left_indices = continue_with_plane(plane_normal, plane_point, left_positive_windows, left_index_tip_pos_stacked_np, left_indices, left_windows, left_gt_keytap_moments, 'left', None)
            right_positions, right_indices = continue_with_plane(plane_normal, plane_point, right_positive_windows, right_index_tip_pos_stacked_np, right_indices, right_windows, right_gt_keytap_moments, 'right', None)

            # TODO Remove
            # left_indices = left_gt_keytap_moments
            # right_indices = right_gt_keytap_moments
            # left_positions = left_points_for_plane_fitting
            # right_positions = right_points_for_plane_fitting
            
            # left_positions, left_indices = continue_with_plane(left_plane_normal, left_plane_point, left_positive_windows, left_index_tip_pos_stacked_np, left_indices, left_windows, left_gt_keytap_moments, 'left', left_inlier_mask)
            # right_positions, right_indices = continue_with_plane(right_plane_normal, right_plane_point, right_positive_windows, right_index_tip_pos_stacked_np, right_indices, right_windows, right_gt_keytap_moments, 'right', right_inlier_mask)

            #np.save(positions_file_to_save, left_positions) # TODO: Remove this if only left needed
            
            
            collated_positions = collate_positions(left_positions, left_indices, right_positions, right_indices)

            # Save the indices as text
            left_indices_str = 'left_indices = [' + ', '.join([str(l) for l in left_indices]) + ']\n'
            right_indices_str = 'right_indices = [' + ', '.join([str(r) for r in right_indices]) + ']'
            with open(indices_file_to_save, 'w+') as f:
                f.write(left_indices_str + right_indices_str)
            
            print('Saved indices to', indices_file_to_save, flush=True)
            
            # Save positions as npy
            np.save(positions_file_to_save, collated_positions)
            print('Saved positions to', positions_file_to_save, flush=True)
    
    # dump whole results
    with open('whole_results_user_side.json', 'w+') as f:
        json.dump(whole_results, f)



if __name__ == '__main__': 

    train = True

    if train:
        main_training(                      \
            train_left=False,               \
            save_left=False,                \
            test_left=True,                \
            train_right=False,              \
            save_right=False,               \
            test_right=True                )
    else:
        main_inference()
