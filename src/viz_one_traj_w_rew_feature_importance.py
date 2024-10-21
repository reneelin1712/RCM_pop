import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import os

# Captum imports for interpretability
from captum.attr import IntegratedGradients, Saliency, FeatureAblation

# Custom modules (adjust the import paths as necessary)
from network_env import RoadWorld
from utils.load_data import (
    load_test_traj,
    ini_od_dist,
    load_path_feature,
    load_link_feature,
    minmax_normalization,
)
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN

def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    gamma = 0.99  # discount factor
    policy_net = PolicyCNN(
        env.n_actions,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        env.pad_idx,
    ).to(device)
    value_net = ValueCNN(
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
    ).to(device)
    discriminator_net = DiscriminatorAIRLCNN(
        env.n_actions,
        gamma,
        env.policy_mask,
        env.state_action,
        path_feature_pad,
        edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
        env.pad_idx,
    ).to(device)

    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discriminator_net.load_state_dict(model_dict['Discrim'])

    policy_net.eval()
    value_net.eval()
    discriminator_net.eval()

    return policy_net, value_net, discriminator_net

def prepare_input_data(env, test_trajs):
    # Select the first trajectory from the test set
    example_traj = test_trajs[0]
    # Get the states and destination from the trajectory
    states = [int(s) for s in example_traj[:-1]]  # All states except the last one
    destination = int(example_traj[-1])  # The destination is the last state
    return states, destination

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
    # Process features to get the CNN input
    input_data = policy_net.process_features(state, des)
    return input_data

def interpret_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature names (ensure this matches the number of channels in your input data)
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
        'Edge ratio',                      #19
        'neighbor mask'
    ]

    # Path settings (adjust paths as necessary)
    model_path = "../trained_models/base/airl_CV0_size10000.pt"
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    test_p = "../data/base/cross_validation/test_CV0.csv"

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
    policy_net, value_net, discriminator_net = load_model(
        model_path, device, env, path_feature_pad, edge_feature_pad
    )

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    # Prepare input data
    states_list, destination = prepare_input_data(env, test_trajs)

    # Ensure output directories exist
    output_dir = 'output_img'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store attributions and importance scores
    attributions_ig_list = []
    attributions_ablation_list = []
    channel_importance_ig_list = []
    channel_importance_ablation_list = []

    disc_attributions_ig_list = []
    disc_attributions_ablation_list = []
    channel_importance_ig_disc_list = []
    channel_importance_ablation_disc_list = []

    rewards_list = []
    actions_list = []

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
            x = input_data
            x = policy_net.forward(x)
            x_mask = policy_net.policy_mask[state_tensor]
            x = x.masked_fill((1 - x_mask).bool(), -1e32)
            action_probs = torch.softmax(x, dim=1)
            return action_probs

        # Integrated Gradients for policy network
        ig = IntegratedGradients(forward_func_policy)
        attributions_ig = ig.attribute(input_data, target=predicted_action)
        attributions_ig_np = attributions_ig.squeeze().cpu().detach().numpy()
        attributions_ig_list.append(attributions_ig_np)

        # Feature Ablation for policy network
        feature_ablation = FeatureAblation(forward_func_policy)
        attributions_ablation = feature_ablation.attribute(input_data, target=predicted_action)
        attributions_ablation_np = attributions_ablation.squeeze().cpu().detach().numpy()
        attributions_ablation_list.append(attributions_ablation_np)

        # Aggregate attributions for policy network
        channel_importance_ig = np.sum(np.abs(attributions_ig_np), axis=(1, 2))
        channel_importance_ig_list.append(channel_importance_ig)

        channel_importance_ablation = np.sum(np.abs(attributions_ablation_np), axis=(1, 2))
        channel_importance_ablation_list.append(channel_importance_ablation)

        # Now compute attributions for the discriminator network
        # Prepare inputs for discriminator
        act_tensor = predicted_action.to(device)
        act_tensor = act_tensor.view(-1).long()
        log_pi_tensor = log_pi.to(device).view(-1)

        # Process features for discriminator
        input_data_disc = discriminator_net.process_neigh_features(state_tensor, des_tensor)
        input_data_disc.requires_grad = True

        def forward_func_discriminator(input_data):
            # Discriminator computations
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
            return f

        # Integrated Gradients for discriminator network
        ig_disc = IntegratedGradients(forward_func_discriminator)
        attributions_ig_disc = ig_disc.attribute(input_data_disc, target=None)
        attributions_ig_disc_np = attributions_ig_disc.squeeze().cpu().detach().numpy()
        disc_attributions_ig_list.append(attributions_ig_disc_np)

        # Feature Ablation for discriminator network
        feature_ablation_disc = FeatureAblation(forward_func_discriminator)
        attributions_ablation_disc = feature_ablation_disc.attribute(input_data_disc, target=None)
        attributions_ablation_disc_np = attributions_ablation_disc.squeeze().cpu().detach().numpy()
        disc_attributions_ablation_list.append(attributions_ablation_disc_np)

        # Aggregate attributions for discriminator network
        channel_importance_ig_disc = np.sum(np.abs(attributions_ig_disc_np), axis=(1, 2))
        channel_importance_ig_disc_list.append(channel_importance_ig_disc)

        channel_importance_ablation_disc = np.sum(np.abs(attributions_ablation_disc_np), axis=(1, 2))
        channel_importance_ablation_disc_list.append(channel_importance_ablation_disc)

        # Calculate reward
        with torch.no_grad():
            reward = discriminator_net.calculate_reward(
                state_tensor, des_tensor, act_tensor, log_pi_tensor, next_state_tensor
            )
            rewards_list.append(reward.item())

    # After the loop, process and visualize the attributions and feature importance rankings
    num_steps = len(channel_importance_ig_list)
    for idx in range(num_steps):
        action = actions_list[idx]

        # Policy Network - Integrated Gradients
        importance_scores = channel_importance_ig_list[idx]
        ranked_indices = np.argsort(-importance_scores)
        sorted_features = [feature_names[i] for i in ranked_indices]
        sorted_importance = importance_scores[ranked_indices]

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': sorted_features,
            'Importance Score': sorted_importance
        })

        # Save DataFrame to CSV
        feature_importance_df.to_csv(
            os.path.join(output_dir, f'policy_ig_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features[::-1], sorted_importance[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Policy Network IG Feature Importance - Step {idx+1} - Action {action}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'policy_ig_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Policy Network - Feature Ablation
        importance_scores_ablation = channel_importance_ablation_list[idx]
        ranked_indices_ablation = np.argsort(-importance_scores_ablation)
        sorted_features_ablation = [feature_names[i] for i in ranked_indices_ablation]
        sorted_importance_ablation = importance_scores_ablation[ranked_indices_ablation]

        # Create DataFrame
        feature_importance_ablation_df = pd.DataFrame({
            'Feature': sorted_features_ablation,
            'Importance Score': sorted_importance_ablation
        })

        # Save DataFrame to CSV
        feature_importance_ablation_df.to_csv(
            os.path.join(output_dir, f'policy_ablation_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features_ablation[::-1], sorted_importance_ablation[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Policy Network Feature Ablation - Step {idx+1} - Action {action}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'policy_ablation_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Discriminator Network - Integrated Gradients
        importance_scores_disc = channel_importance_ig_disc_list[idx]
        ranked_indices_disc = np.argsort(-importance_scores_disc)
        sorted_features_disc = [feature_names[i] for i in ranked_indices_disc]
        sorted_importance_disc = importance_scores_disc[ranked_indices_disc]

        # Create DataFrame
        feature_importance_disc_df = pd.DataFrame({
            'Feature': sorted_features_disc,
            'Importance Score': sorted_importance_disc
        })

        # Save DataFrame to CSV
        feature_importance_disc_df.to_csv(
            os.path.join(output_dir, f'discriminator_ig_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features_disc[::-1], sorted_importance_disc[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Discriminator IG Feature Importance - Step {idx+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'discriminator_ig_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Discriminator Network - Feature Ablation
        importance_scores_ablation_disc = channel_importance_ablation_disc_list[idx]
        ranked_indices_ablation_disc = np.argsort(-importance_scores_ablation_disc)
        sorted_features_ablation_disc = [feature_names[i] for i in ranked_indices_ablation_disc]
        sorted_importance_ablation_disc = importance_scores_ablation_disc[ranked_indices_ablation_disc]

        # Create DataFrame
        feature_importance_ablation_disc_df = pd.DataFrame({
            'Feature': sorted_features_ablation_disc,
            'Importance Score': sorted_importance_ablation_disc
        })

        # Save DataFrame to CSV
        feature_importance_ablation_disc_df.to_csv(
            os.path.join(output_dir, f'discriminator_ablation_feature_importance_step_{idx+1}.csv'), index=False
        )

        # Plot Feature Importance
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features_ablation_disc[::-1], sorted_importance_ablation_disc[::-1])
        plt.xlabel('Importance Score')
        plt.title(f'Discriminator Feature Ablation - Step {idx+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'discriminator_ablation_feature_importance_step_{idx+1}.png'))
        plt.close()

        # Plot Heatmaps for Policy Network
        # Integrated Gradients
        attr_ig_sum = np.sum(attributions_ig_list[idx], axis=0)
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ig_sum, cmap='hot')
        plt.title(f'Policy Network IG Attribution Map - Step {idx+1} - Action {action}')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'policy_ig_heatmap_step_{idx+1}.png'))
        plt.close()

        # Feature Ablation
        attr_ablation_sum = np.sum(attributions_ablation_list[idx], axis=0)
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ablation_sum, cmap='hot')
        plt.title(f'Policy Network Feature Ablation Attribution Map - Step {idx+1} - Action {action}')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'policy_ablation_heatmap_step_{idx+1}.png'))
        plt.close()

        # Plot Heatmaps for Discriminator Network
        # Integrated Gradients
        attr_ig_disc_sum = np.sum(disc_attributions_ig_list[idx], axis=0)
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ig_disc_sum, cmap='hot')
        plt.title(f'Discriminator IG Attribution Map - Step {idx+1}')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'discriminator_ig_heatmap_step_{idx+1}.png'))
        plt.close()

        # Feature Ablation
        attr_ablation_disc_sum = np.sum(disc_attributions_ablation_list[idx], axis=0)
        plt.figure(figsize=(6, 4))
        plt.imshow(attr_ablation_disc_sum, cmap='hot')
        plt.title(f'Discriminator Feature Ablation Attribution Map - Step {idx+1}')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'discriminator_ablation_heatmap_step_{idx+1}.png'))
        plt.close()

        # Optionally, visualize attributions for each channel (Policy Network - Integrated Gradients)
        num_channels = attributions_ig_list[idx].shape[0]
        for i in range(num_channels):
            plt.figure(figsize=(6, 4))
            plt.imshow(attributions_ig_list[idx][i], cmap='hot')
            plt.title(f'Policy IG - {feature_names[i]} - Step {idx+1} - Action {action}')
            plt.colorbar()
            plt.savefig(
                os.path.join(output_dir, f'policy_ig_channel_{i}_step_{idx+1}.png')
            )
            plt.close()

    # Plot the rewards over the trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(rewards_list) + 1), rewards_list, marker='o')
    plt.title('Rewards over Trajectory Steps')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rewards_over_trajectory.png'))
    plt.close()

    print("Interpretation complete. Results saved in the 'output_img' directory.")

if __name__ == "__main__":
    interpret_model()
