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
import matplotlib.pyplot as plt
# Compute attributions
from captum.attr import IntegratedGradients, Saliency


def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    gamma = 0.99  # discount factor
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

    model_dict = torch.load(model_path, map_location=device,weights_only=True)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    policy_net.eval()  # Set to evaluation mode
    value_net.eval()
    discrim_net.eval()

    return policy_net, value_net, discrim_net


def prepare_input_data(env, test_trajs):
    # For simplicity, select the first trajectory from the test set
    example_traj = test_trajs[0]
    # Get the states and destination from the trajectory
    states = [int(s) for s in example_traj[:-1]]  # All states except the last one
    destination = int(example_traj[-1])  # The destination is the last state
    return states, destination


def get_cnn_input(policy_net, state, des,device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Path settings
    model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = "../data/base/cross_validation/test_CV0.csv"  # Adjust path and CV index as necessary

    # Initialize environment
    od_list, od_dist = ini_od_dist(test_p)
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

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Prepare input data
    states_list, destination = prepare_input_data(env, test_trajs)

    # Initialize lists to store attributions
    attributions_ig_list = []
    attributions_saliency_list = []

    # Loop over the states in the trajectory
    for state in states_list:
        # Get CNN input
        input_data = get_cnn_input(policy_net, state, destination, device)
        input_data.requires_grad = True

        # Need to keep 'state' and 'des' in scope for the forward function
        state_tensor = torch.tensor([state], dtype=torch.long).to(device)
        des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

        # Define the forward function
        def forward_func(input_data):
            x = input_data
            x = policy_net.forward(x)
            x_mask = policy_net.policy_mask[state_tensor]
            x = x.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(x, dim=1)
            return action_probs

        with torch.no_grad():
            output = forward_func(input_data)
        predicted_action = torch.argmax(output, dim=1)

        # Integrated Gradients
        ig = IntegratedGradients(forward_func)
        attributions_ig = ig.attribute(input_data, target=predicted_action)

        # Saliency
        saliency = Saliency(forward_func)
        attributions_saliency = saliency.attribute(input_data, target=predicted_action)

        # Store the attributions
        attributions_ig_list.append(attributions_ig.squeeze().cpu().detach().numpy())
        attributions_saliency_list.append(attributions_saliency.squeeze().cpu().detach().numpy())
    

    # Visualize attributions for each step
    for idx, (attr_ig, attr_saliency) in enumerate(zip(attributions_ig_list, attributions_saliency_list)):
        # Sum over the channels to get a single 2D map
        attr_ig_sum = np.sum(attr_ig, axis=0)
        attr_saliency_sum = np.sum(attr_saliency, axis=0)

        # Plot the Integrated Gradients attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ig_sum, cmap='hot')
        plt.title(f'Integrated Gradients Attribution - Step {idx+1}')
        plt.colorbar()
        plt.show()

        # Plot the Saliency attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_saliency_sum, cmap='hot')
        plt.title(f'Saliency Attribution - Step {idx+1}')
        plt.colorbar()
        plt.show()

        # Visualize attributions for each channel
        num_channels = attr_ig.shape[0]
        for i in range(num_channels):
            plt.figure(figsize=(6, 4))
            plt.imshow(attr_ig[i], cmap='hot')
            plt.title(f'Integrated Gradients Attribution - Channel {i}')
            plt.colorbar()
            plt.show()


interpret_model()