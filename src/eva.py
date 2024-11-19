import numpy as np
import torch
from network_env import RoadWorld
from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from core.agent import Agent
from utils.evaluation import evaluate_model
import pandas as pd



def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    # Assuming the dimensions for the models based on training setup
    gamma = 0.95  # discount factor
    edge_data = pd.read_csv('../data/base/edge.txt')
    policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                           path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                       env.state_action, path_feature_pad, edge_feature_pad,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                       env.pad_idx).to(device)

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    return policy_net, value_net, discrim_net

# original not evaluate on training data
# def evaluate_only():
#     device = torch.device('cpu')
    
#     # Path settings
#     model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
#     edge_p = "../data/base/edge.txt"
#     network_p = "../data/base/transit.npy"
#     path_feature_p = "../data/base/feature_od.npy"
#     test_p = "../data/base/cross_validation/test_CV0.csv"  # Adjust path and CV index as necessary

#     # Initialize environment
#     od_list, od_dist = ini_od_dist(test_p)
#     env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))

#     # Load features and normalize
#     path_feature, path_max, path_min = load_path_feature(path_feature_p)
#     edge_feature, link_max, link_min = load_link_feature(edge_p)
#     path_feature = minmax_normalization(path_feature, path_max, path_min)
#     path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
#     path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
#     edge_feature = minmax_normalization(edge_feature, link_max, link_min)
#     edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
#     edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

#     # Load the model
#     policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

#     # Load test trajectories
#     test_trajs, test_od = load_test_traj(test_p)

#     # Evaluate the model
#     evaluate_model(test_od, test_trajs, policy_net, env)


# not calculating on the export data
# def evaluate_only(dataset='test'):
#     device = torch.device('cpu')
    
#     # Path settings
#     model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
#     edge_p = "../data/base/edge.txt"
#     network_p = "../data/base/transit.npy"
#     path_feature_p = "../data/base/feature_od.npy"
#     if dataset == 'test':
#         data_p = "../data/base/cross_validation/test_CV0.csv"  # Adjust path and CV index as necessary
#     elif dataset == 'train':
#         data_p = "../data/base/cross_validation/train_CV0_size10000.csv"  # Adjust path and CV index as necessary
#     else:
#         raise ValueError("Invalid dataset choice. Use 'train' or 'test'.")
    
#     # Initialize environment
#     od_list, od_dist = ini_od_dist(data_p)
#     env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    
#     # Load features and normalize
#     path_feature, path_max, path_min = load_path_feature(path_feature_p)
#     edge_feature, link_max, link_min = load_link_feature(edge_p)
#     path_feature = minmax_normalization(path_feature, path_max, path_min)
#     path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
#     path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
#     edge_feature = minmax_normalization(edge_feature, link_max, link_min)
#     edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
#     edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    
#     # Load the model
#     policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)
    
#     # Load trajectories
#     test_trajs, test_od = load_test_traj(data_p)
    
#     # Evaluate the model
#     evaluate_model(test_od, test_trajs, policy_net, env)

def evaluate_only():
    device = torch.device('cpu')
    
    # Path settings
    # model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
    model_path = "../trained_models/base/ed005.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    train_p = "../data/base/cross_validation/train_CV0_size10000.csv"
    test_p = "../data/base/cross_validation/test_CV0.csv"
    
    # Initialize environment
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    
    # Load features and normalize
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature
    
    # Load the model
    policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)
    
    # Evaluate on Training Data
    print('Evaluating on Training Data...')
    # TODO
    train_trajs, train_od = load_test_traj(train_p)  # load_train_sample 
    # Evaluate on training data
    # evaluate_model("train", train_od, train_trajs, policy_net, env)
    
    # Evaluate on Test Data
    print('Evaluating on Test Data...')
    test_trajs, test_od = load_test_traj(test_p)
    # evaluate_model(test_od, test_trajs, policy_net, env)
    # Evaluate on test data
    evaluate_model("test", test_od, test_trajs, policy_net, env)







if __name__ == '__main__':
    evaluate_only()
    # print('Evaluating on Training Data...')
    # evaluate_only(dataset='train')
    # print('Evaluating on Test Data...')
    # evaluate_only(dataset='test')