import os
import numpy as np
import torch
import torch.nn.functional as F
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
import matplotlib.pyplot as plt
import pandas as pd
from captum.attr import IntegratedGradients, Saliency

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

def get_cnn_input(policy_net, state, des, device):
    state = torch.tensor([state], dtype=torch.long).to(device)
    des = torch.tensor([des], dtype=torch.long).to(device)
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
        'Edge ratio',                     #19
        'Neighbor mask'                   #20
    ]

    # Path settings
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

    # Ensure output directories exist
    output_dir = 'output_img/attribution'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize variables to store attributions and rewards
    total_attr_ig_policy = None
    total_attr_saliency_policy = None
    total_attr_ig_discrim = None
    total_attr_saliency_discrim = None
    total_channel_importance_policy = None
    total_channel_importance_discrim = None
    rewards_over_steps = []
    counts_per_step = []
    max_steps = max(len(traj) - 1 for traj in test_trajs)
    num_features = len(feature_names)
    count = 0

    for step_idx in range(max_steps):
        rewards_over_steps.append([])
        counts_per_step.append(0)

    for traj_idx, traj in enumerate(test_trajs):
        states_list = [int(s) for s in traj[:-1]]
        destination = int(traj[-1])

        for idx, state in enumerate(states_list):
            # Prepare policy network input
            input_data = get_cnn_input(policy_net, state, destination, device)
            input_data.requires_grad = True

            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

            # Define the forward function for the policy network
            def forward_func_policy(input_data):
                x = policy_net.forward(input_data)
                x_mask = policy_net.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs

            with torch.no_grad():
                output = forward_func_policy(input_data)
            predicted_action = torch.argmax(output, dim=1)

            # Compute attributions for the policy network
            ig = IntegratedGradients(forward_func_policy)
            attributions_ig = ig.attribute(input_data, target=predicted_action)

            attr_ig_np = attributions_ig.squeeze().cpu().detach().numpy()

            # Compute channel importance
            channel_importance = np.sum(np.abs(attr_ig_np), axis=(1, 2))

            if total_attr_ig_policy is None:
                total_attr_ig_policy = attr_ig_np
                total_channel_importance_policy = channel_importance
            else:
                total_attr_ig_policy += attr_ig_np
                total_channel_importance_policy += channel_importance

            # Now prepare discriminator network input
            # Prepare inputs for discriminator
            act_tensor = predicted_action.to(device).view(-1).long()
            log_pi_tensor = torch.log(output[0, predicted_action]).to(device).view(-1)

            # Get next state
            action = predicted_action.item()
            next_state = env.state_action[state][action]
            next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)

            # Process features for discriminator
            input_data_disc = discriminator_net.process_neigh_features(state_tensor, des_tensor)
            input_data_disc.requires_grad = True

            # Define the forward function for the discriminator
            def forward_func_discriminator(input_data_disc):
                x = discriminator_net.pool(F.leaky_relu(discriminator_net.conv1(input_data_disc), 0.2))
                x = F.leaky_relu(discriminator_net.conv2(x), 0.2)
                x = x.view(-1, 30)

                x_act = F.one_hot(act_tensor, num_classes=discriminator_net.action_num).to(device)
                if x_act.dim() == 1:
                    x_act = x_act.unsqueeze(0)
                batch_size = x.shape[0]
                x_act = x_act.expand(batch_size, -1)

                x = torch.cat([x, x_act], 1)

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

            # Compute attributions for the discriminator network
            ig_disc = IntegratedGradients(forward_func_discriminator)
            attributions_ig_disc = ig_disc.attribute(input_data_disc, target=None)

            attr_ig_disc_np = attributions_ig_disc.squeeze().cpu().detach().numpy()

            # Compute channel importance for discriminator
            channel_importance_disc = np.sum(np.abs(attr_ig_disc_np), axis=(1, 2))

            if total_attr_ig_discrim is None:
                total_attr_ig_discrim = attr_ig_disc_np
                total_channel_importance_discrim = channel_importance_disc
            else:
                total_attr_ig_discrim += attr_ig_disc_np
                total_channel_importance_discrim += channel_importance_disc

            # Calculate reward
            with torch.no_grad():
                reward = discriminator_net.calculate_reward(
                    state_tensor, des_tensor, act_tensor, log_pi_tensor, next_state_tensor
                )
                # Collect rewards per step
                rewards_over_steps[idx].append(reward.item())
                counts_per_step[idx] += 1

            count += 1

    # Compute average attributions for policy network
    avg_attr_ig_policy = total_attr_ig_policy / count
    avg_channel_importance_policy = total_channel_importance_policy / count

    # Compute average attributions for discriminator network
    avg_attr_ig_discrim = total_attr_ig_discrim / count
    avg_channel_importance_discrim = total_channel_importance_discrim / count

    # Sum over the spatial dimensions to get feature importance
    policy_feature_importance = avg_channel_importance_policy
    discrim_feature_importance = avg_channel_importance_discrim

    # Rank features for policy network
    ranked_indices_policy = np.argsort(-policy_feature_importance)
    sorted_features_policy = [feature_names[i] for i in ranked_indices_policy]
    sorted_importance_policy = policy_feature_importance[ranked_indices_policy]

    # Rank features for discriminator network
    ranked_indices_discrim = np.argsort(-discrim_feature_importance)
    sorted_features_discrim = [feature_names[i] for i in ranked_indices_discrim]
    sorted_importance_discrim = discrim_feature_importance[ranked_indices_discrim]

    # Create DataFrame for policy network
    feature_importance_policy_df = pd.DataFrame({
        'Feature': sorted_features_policy,
        'Importance Score': sorted_importance_policy
    })

    # Save DataFrame to CSV
    feature_importance_policy_df.to_csv(
        os.path.join(output_dir, 'policy_feature_importance.csv'), index=False
    )

    # Plot Feature Importance for policy network
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features_policy[::-1], sorted_importance_policy[::-1])
    plt.xlabel('Importance Score')
    plt.title('Policy Network Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_feature_importance.png'))
    plt.close()

    # Create DataFrame for discriminator network
    feature_importance_discrim_df = pd.DataFrame({
        'Feature': sorted_features_discrim,
        'Importance Score': sorted_importance_discrim
    })

    # Save DataFrame to CSV
    feature_importance_discrim_df.to_csv(
        os.path.join(output_dir, 'discriminator_feature_importance.csv'), index=False
    )

    # Plot Feature Importance for discriminator network
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features_discrim[::-1], sorted_importance_discrim[::-1])
    plt.xlabel('Importance Score')
    plt.title('Discriminator Network Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_feature_importance.png'))
    plt.close()

    # Compute average rewards over steps
    avg_rewards = []
    steps = []
    for idx, rewards in enumerate(rewards_over_steps):
        if counts_per_step[idx] > 0:
            avg_reward = np.mean(rewards)
            avg_rewards.append(avg_reward)
            steps.append(idx + 1)  # Step index starts from 1
        else:
            break  # No more steps with rewards

    # Plot the average rewards over trajectory steps
    plt.figure(figsize=(6, 4))
    plt.plot(steps, avg_rewards, marker='o')
    plt.title('Average Rewards over Trajectory Steps')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_rewards_over_trajectory.png'))
    plt.close()

    print("Interpretation complete. Results saved in the 'output_img/attribution' directory.")

if __name__ == "__main__":
    interpret_model()
