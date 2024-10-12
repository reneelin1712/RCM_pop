import numpy as np
import torch
from network_env import RoadWorld
from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency
import torch.nn.functional as F
import json
import os

def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    gamma = 0.99  # discount factor
    policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                           path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    discriminator_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                             env.state_action, path_feature_pad, edge_feature_pad,
                                             path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                             path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                             env.pad_idx).to(device)

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discriminator_net.load_state_dict(model_dict['Discrim'])

    policy_net.eval()
    value_net.eval()
    discriminator_net.eval()

    return policy_net, value_net, discriminator_net

def prepare_input_data(env, test_trajs):
    # For simplicity, select the first trajectory from the test set
    example_traj = test_trajs[0]
    # Get the states and destination from the trajectory
    states = [int(s) for s in example_traj[:-1]]  # All states except the last one
    destination = int(example_traj[-1])  # The destination is the last state
    return states, destination

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_names = [
    # Path features (12 features)
    'Number of links',                # 0
    'Total length',                   # 1
    'Number of left turns',           # 2
    'Number of right turns',          # 3
    'Number of U-turns',              # 4
    'Number of residential roads',    # 5
    'Number of primary roads',        # 6
    'Number of unclassified roads',   # 7
    'Number of tertiary roads',       # 8
    'Number of living_street roads',  # 9
    'Number of secondary roads',      #10
    'Mask feature',                   #11
    # Edge features (8 features)
    'Edge length',                    #12
    'Highway type: residential',      #13
    'Highway type: primary',          #14
    'Highway type: unclassified',     #15
    'Highway type: tertiary',         #16
    'Highway type: living_street',    #17
    'Highway type: secondary',        #18
    'Edge ratio'                      #19
]

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
    policy_net, value_net, discriminator_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Prepare input data
    states_list, destination = prepare_input_data(env, test_trajs)

    # Initialize lists to store attributions
    attributions_ig_list = []
    attributions_saliency_list = []
    disc_attributions_ig_list = []
    disc_attributions_saliency_list = []
    rewards_list = []

    actions_list = []  # Initialize at the beginning of the function

    # Loop over the states in the trajectory
    for idx, state in enumerate(states_list):
        # Get CNN input for policy network
        input_data = get_cnn_input(policy_net, state, destination, device)
        input_data.requires_grad = True

        # Prepare tensors
        state_tensor = torch.tensor([state], dtype=torch.long).to(device)
        des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

        # Get action probabilities from policy network
        with torch.no_grad():
            output = policy_net.forward(input_data)
            x_mask = policy_net.policy_mask[state_tensor]
            output = output.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(output, dim=1)
            predicted_action = torch.argmax(action_probs, dim=1)
            log_pi = torch.log(action_probs[0, predicted_action])

        # Get next state
        action = predicted_action.item()
        actions_list.append(action)
        next_state = env.state_action[state][action]
        next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

        # Compute attributions for policy network
        def forward_func_policy(input_data):
            x = policy_net.forward(input_data)
            x_mask = policy_net.policy_mask[state_tensor]
            x = x.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(x, dim=1)
            return action_probs

        # Integrated Gradients for policy network
        ig = IntegratedGradients(forward_func_policy)
        attributions_ig = ig.attribute(input_data, target=predicted_action)

        # Saliency for policy network
        saliency = Saliency(forward_func_policy)
        attributions_saliency = saliency.attribute(input_data, target=predicted_action)

        # Store the attributions
        attributions_ig_list.append(attributions_ig.squeeze().cpu().detach().numpy())
        attributions_saliency_list.append(attributions_saliency.squeeze().cpu().detach().numpy())


        # Now compute attributions for the discriminator network

        # Prepare inputs for discriminator
        act_tensor = predicted_action.to(device)
        # Ensure act_tensor is a 1D LongTensor
        act_tensor = act_tensor.view(-1).long()

        # log_pi_tensor = log_pi.to(device)
        log_pi_tensor = log_pi.to(device).view(-1)  # Ensure shape [batch_size]


        # Process features for discriminator
        input_data_disc = discriminator_net.process_neigh_features(state_tensor, des_tensor)
        input_data_disc.requires_grad = True

        # Define the forward function for discriminator
        def forward_func_discriminator(input_data):
            # The discriminator uses both the current and next states
            # We'll process state features separately
            x = input_data
            x = discriminator_net.pool(F.leaky_relu(discriminator_net.conv1(x), 0.2))
            x = F.leaky_relu(discriminator_net.conv2(x), 0.2)
            x = x.view(-1, 30)  # x shape: [batch_size, 30]

            # Compute x_act
            x_act = F.one_hot(act_tensor, num_classes=discriminator_net.action_num).to(device)
            if x_act.dim() == 1:
                x_act = x_act.unsqueeze(0)  # x_act shape: [1, num_classes]

            # Expand x_act to match the batch size of x
            batch_size = x.shape[0]
            x_act = x_act.expand(batch_size, -1)  # x_act shape: [batch_size, num_classes]

            # Concatenate x and x_act
            x = torch.cat([x, x_act], 1)  # x shape: [batch_size, 30 + num_classes]

            x = F.leaky_relu(discriminator_net.fc1(x), 0.2)
            x = F.leaky_relu(discriminator_net.fc2(x), 0.2)
            rs = discriminator_net.fc3(x)

            # Compute hs and hs_next
            x_state = discriminator_net.process_state_features(state_tensor, des_tensor)
            x_state = F.leaky_relu(discriminator_net.h_fc1(x_state), 0.2)
            x_state = F.leaky_relu(discriminator_net.h_fc2(x_state), 0.2)
            x_state = discriminator_net.h_fc3(x_state)

            next_x_state = discriminator_net.process_state_features(next_state_tensor, des_tensor)
            next_x_state = F.leaky_relu(discriminator_net.h_fc1(next_x_state), 0.2)
            next_x_state = F.leaky_relu(discriminator_net.h_fc2(next_x_state), 0.2)
            next_x_state = discriminator_net.h_fc3(next_x_state)

            f = rs + discriminator_net.gamma * next_x_state - x_state
            # Output for attribution
            return f

        # Compute attributions for the discriminator
        ig_disc = IntegratedGradients(forward_func_discriminator)
        attributions_ig_disc = ig_disc.attribute(input_data_disc, target=None)

        saliency_disc = Saliency(forward_func_discriminator)
        attributions_saliency_disc = saliency_disc.attribute(input_data_disc, target=None)

        # Store attributions
        disc_attributions_ig_list.append(attributions_ig_disc.squeeze().cpu().detach().numpy())
        disc_attributions_saliency_list.append(attributions_saliency_disc.squeeze().cpu().detach().numpy())

        # Calculate reward
        with torch.no_grad():
            # reward = discriminator_net.calculate_reward(state_tensor, des_tensor, act_tensor.unsqueeze(0), log_pi_tensor.unsqueeze(0), next_state_tensor)
            reward = discriminator_net.calculate_reward(state_tensor, des_tensor, act_tensor, log_pi_tensor, next_state_tensor)
            rewards_list.append(reward.item())

    # Visualize attributions for each step
    for idx, (attr_ig, attr_saliency, attr_ig_disc, attr_saliency_disc) in enumerate(zip(
            attributions_ig_list, attributions_saliency_list, disc_attributions_ig_list, disc_attributions_saliency_list)):
        # Sum over the channels to get a single 2D map
        attr_ig_sum = np.sum(attr_ig, axis=0)
        attr_saliency_sum = np.sum(attr_saliency, axis=0)
        attr_ig_disc_sum = np.sum(attr_ig_disc, axis=0)
        attr_saliency_disc_sum = np.sum(attr_saliency_disc, axis=0)

        # Retrieve the action for the current step
        action = actions_list[idx]

        # Plot the Integrated Gradients attribution for policy network
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ig_sum, cmap='hot')
        
        # Modify the titles to include the action
        plt.title(f'Policy Network IG Attribution - Step {idx+1} - Action {action}')
        plt.colorbar()
        plt.savefig(os.path.join('output_img', f'ig_heatmap_step_{idx+1}.png'))
        plt.show()

        # Plot the Saliency attribution for policy network
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_saliency_sum, cmap='hot')
        plt.title(f'Policy Network Saliency Attribution - Step {idx+1} - Action {action}')
        plt.colorbar()
        plt.show()

        # Plot the Integrated Gradients attribution for discriminator network
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ig_disc_sum, cmap='hot')
        plt.title(f'Discriminator IG Attribution - Step {idx+1}')
        plt.colorbar()
        plt.show()

        # Plot the Saliency attribution for discriminator network
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_saliency_disc_sum, cmap='hot')
        plt.title(f'Discriminator Saliency Attribution - Step {idx+1}')
        plt.colorbar()
        plt.show()

        # Visualize attributions for each channel with feature names
        num_channels = attr_ig.shape[0]
        for i in range(num_channels):
            plt.figure(figsize=(6, 4))
            plt.imshow(attr_ig[i], cmap='hot')
            plt.title(f'Policy IG - {feature_names[i]} - Step {idx+1} - Action {action}')
            plt.colorbar()
            plt.show()

    # Plot the rewards over the trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rewards_list)+1), rewards_list)
    plt.title('Rewards over Trajectory Steps')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.show()

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

interpret_model()
