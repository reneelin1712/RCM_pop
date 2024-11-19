import pandas as pd
from collections import Counter
import torch
import numpy as np
import csv
import os

def analyze_dataset(data_p):
    # Read the dataset
    df = pd.read_csv(data_p)  # Adjust separator if needed

    # Ensure 'path' is a string
    df['path'] = df['path'].astype(str)

    # Group by OD pair
    od_group = df.groupby(['ori', 'des'])

    # Prepare data for analysis
    od_analysis = []
    total_trajectories = len(df)

    # This list will hold per-path data
    per_path_data = []

    for od, group in od_group:
        ori, des = od
        total_count = len(group)
        frequency = total_count / total_trajectories
        unique_paths = group['path'].unique()
        num_unique_paths = len(unique_paths)

        od_analysis.append({
            'origin': ori,
            'destination': des,
            'total_count': total_count,
            'frequency': frequency,
            'num_unique_paths': num_unique_paths,
            'unique_paths': list(unique_paths)
        })

        # Now, for each unique path, calculate its count and frequency within this OD pair
        path_counts = group['path'].value_counts().to_dict()
        for path, count in path_counts.items():
            path_frequency = count / total_trajectories  # Overall frequency
            path_frequency_within_od = count / total_count  # Frequency within this OD pair
            per_path_data.append({
                'origin': ori,
                'destination': des,
                'path': path,
                'path_count': count,
                'path_frequency': path_frequency,
                'path_frequency_within_od': path_frequency_within_od,
                'total_count': total_count,
                'od_frequency': frequency,
            })

    # Create a DataFrame for analysis
    df_analysis = pd.DataFrame(od_analysis)

    # Save the first file: All unique OD trajectories with counts and frequencies
    df_analysis.to_csv('./eva/od_analysis_all.csv', index=False)

    # Save the second file: OD trajectories with more than one kind of route
    df_multiple_routes = df_analysis[df_analysis['num_unique_paths'] > 1]
    df_multiple_routes.to_csv('./eva/od_analysis_multiple_routes.csv', index=False)

    # Save the third file: OD trajectories with only one kind of route
    df_single_route = df_analysis[df_analysis['num_unique_paths'] == 1]
    df_single_route.to_csv('./eva/od_analysis_single_route.csv', index=False)

    # Now, create a DataFrame for per-path data
    df_per_path = pd.DataFrame(per_path_data)
    # Save per-path data
    df_per_path.to_csv('./eva/od_analysis_per_path.csv', index=False)

    print("Analysis files saved:")
    print("1. od_analysis_all.csv")
    print("2. od_analysis_multiple_routes.csv")
    print("3. od_analysis_single_route.csv")
    print("4. od_analysis_per_path.csv")

    return df_analysis, df_multiple_routes, df_single_route, df_per_path

def generate_trajectories_per_path(df_per_path, model, env, device, max_steps=50):
    generated_trajs = []

    for idx, row in df_per_path.iterrows():
        ori = row['origin']
        des = row['destination']
        path_count = int(row['path_count'])
        path = row['path']

        # Generate the same number of trajectories as 'path_count'
        for _ in range(path_count):
            # Reset environment for the current OD pair
            curr_ori, curr_des = env.reset(st=ori, des=des)
            sample_path = [str(curr_ori)]
            actions_taken = []
            curr_state = curr_ori

            for _ in range(max_steps):
                state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
                des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)

                # Get action probabilities from the model
                with torch.no_grad():
                    action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
                    action_probs_np = action_probs.cpu().numpy()

                # Sample an action based on probabilities
                action = np.random.choice(len(action_probs_np), p=action_probs_np)

                # Take a step in the environment
                next_state, reward, done = env.step(action)

                # Append the action and the next state to the path
                actions_taken.append(int(action))  # Ensure action is an integer
                if next_state != env.pad_idx:
                    sample_path.append(str(next_state))
                curr_state = next_state

                if done:
                    break

            # Append the generated trajectory and associated information
            generated_trajs.append({
                'origin': ori,
                'destination': des,
                'dataset_path': path,
                'generated_trajectory': '_'.join(sample_path),
                'trajectory_length': len(sample_path),
                'actions_taken': actions_taken,
                'path_count': path_count
            })

    return generated_trajs

def compute_rewards_for_generated_trajs(generated_trajs, env, discriminator_net, device):
    # Updated list to store the trajectories with rewards
    trajectories_with_rewards = []

    for traj_info in generated_trajs:
        ori = traj_info['origin']
        des = traj_info['destination']
        traj_str = traj_info['generated_trajectory']
        actions_taken = traj_info['actions_taken']
        sample_path = traj_str.split('_')
        sample_path = [int(s) for s in sample_path]  # Convert to integers

        rewards_list = []
        cumulative_reward = 0.0

        for idx in range(len(sample_path) - 1):
            state = sample_path[idx]
            next_state = sample_path[idx + 1]
            action = actions_taken[idx]

            # Prepare tensors
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            next_state_tensor = torch.tensor([next_state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([des], dtype=torch.long).to(device)
            act_tensor = torch.tensor([action], dtype=torch.long).to(device)

            # Get log_pi (assuming policy is deterministic here)
            log_pi = torch.tensor([0.0], dtype=torch.float).to(device)

            # Compute reward using the discriminator network
            with torch.no_grad():
                reward = discriminator_net.calculate_reward(
                    state_tensor, des_tensor, act_tensor, log_pi, next_state_tensor
                )
                reward = reward.item()
                rewards_list.append(reward)
                cumulative_reward += reward

        # Find highest and lowest reward step index and value
        rewards_array = np.array(rewards_list)
        highest_reward_index = np.argmax(rewards_array)
        highest_reward_value = rewards_array[highest_reward_index]

        lowest_reward_index = np.argmin(rewards_array)
        lowest_reward_value = rewards_array[lowest_reward_index]

        # Add rewards and cumulative reward to the trajectory info
        traj_info['rewards_per_step'] = rewards_list
        traj_info['total_return'] = cumulative_reward
        traj_info['highest_reward_index'] = highest_reward_index + 1  # Steps are 1-indexed
        traj_info['highest_reward_value'] = highest_reward_value
        traj_info['lowest_reward_index'] = lowest_reward_index + 1  # Steps are 1-indexed
        traj_info['lowest_reward_value'] = lowest_reward_value

        # Append updated trajectory info to the list
        trajectories_with_rewards.append(traj_info)

    return trajectories_with_rewards

def identify_divergence_and_convergence_points(trajs_with_rewards):
    # Group trajectories by OD pair
    grouped = {}
    for traj in trajs_with_rewards:
        od_pair = (traj['origin'], traj['destination'])
        if od_pair not in grouped:
            grouped[od_pair] = []
        grouped[od_pair].append(traj)

    # For each OD pair, identify divergence and convergence points
    for od_pair, traj_list in grouped.items():
        # Need at least two trajectories to compare
        if len(traj_list) < 2:
            continue

        # Get the minimum length among trajectories based on rewards_per_step
        min_length = min(len(traj['rewards_per_step']) for traj in traj_list)
        # Align trajectories
        aligned_trajs = []
        for traj in traj_list:
            # Since rewards_per_step has length N, aligned_path should have length N + 1
            path = traj['generated_trajectory'].split('_')[:min_length + 1]
            traj['aligned_path'] = path
            aligned_trajs.append(traj)

        # Identify divergence point
        divergence_index = None
        for idx in range(min_length):
            states_at_step = set(traj['aligned_path'][idx] for traj in aligned_trajs)
            if len(states_at_step) > 1:
                divergence_index = idx
                break

        # Identify convergence point
        convergence_index = None
        if divergence_index is not None:
            for idx in range(divergence_index + 1, min_length + 1):
                states_at_step = set(traj['aligned_path'][idx] for traj in aligned_trajs)
                if len(states_at_step) == 1:
                    convergence_index = idx
                    break

        # Record divergence and convergence information
        for traj in aligned_trajs:
            # Divergence information
            if divergence_index is not None and divergence_index < len(traj['rewards_per_step']):
                traj['divergence_index'] = divergence_index + 1
                traj['divergence_state'] = traj['aligned_path'][divergence_index]
                traj['divergence_reward'] = traj['rewards_per_step'][divergence_index]
            else:
                traj['divergence_index'] = None
                traj['divergence_state'] = None
                traj['divergence_reward'] = None

            # Convergence information
            if convergence_index is not None and convergence_index < len(traj['rewards_per_step']):
                traj['convergence_index'] = convergence_index + 1
                traj['convergence_state'] = traj['aligned_path'][convergence_index]
                traj['convergence_reward'] = traj['rewards_per_step'][convergence_index]
            else:
                traj['convergence_index'] = None
                traj['convergence_state'] = None
                traj['convergence_reward'] = None

    return trajs_with_rewards

def identify_divergence_points(trajs_with_rewards):
    # Group trajectories by OD pair
    grouped = {}
    for traj in trajs_with_rewards:
        od_pair = (traj['origin'], traj['destination'])
        if od_pair not in grouped:
            grouped[od_pair] = []
        grouped[od_pair].append(traj)

    # For each OD pair, identify divergence points
    for od_pair, traj_list in grouped.items():
        # Get the minimum length among trajectories based on rewards_per_step
        min_length = min(len(traj['rewards_per_step']) for traj in traj_list)
        # Align trajectories
        aligned_trajs = []
        for traj in traj_list:
            # Since rewards_per_step has length N, aligned_path should have length N + 1
            path = traj['generated_trajectory'].split('_')[:min_length + 1]
            traj['aligned_path'] = path
            aligned_trajs.append(traj)

        # Identify divergence points
        for idx in range(min_length):
            states_at_step = set(traj['aligned_path'][idx] for traj in aligned_trajs)
            if len(states_at_step) > 1:
                # Divergence occurs at this index
                divergence_index = idx
                break
        else:
            divergence_index = None  # No divergence found

        # Record divergence information
        for traj in aligned_trajs:
            if divergence_index is not None:
                traj['divergence_index'] = divergence_index + 1  # Steps are 1-indexed
                traj['divergence_state'] = traj['aligned_path'][divergence_index]
                traj['divergence_reward'] = traj['rewards_per_step'][divergence_index]
            else:
                traj['divergence_index'] = None
                traj['divergence_state'] = None
                traj['divergence_reward'] = None

    return trajs_with_rewards


def save_generated_trajectories_matched(generated_trajs_with_rewards, filename):
    # Convert the list of dictionaries to a DataFrame
    df_generated = pd.DataFrame(generated_trajs_with_rewards)

    # Save to CSV
    df_generated.to_csv(filename, index=False)
    print(f"Generated trajectories with rewards saved to {filename}")

def main():
    device = torch.device('cpu')  # or 'cuda' if using GPU

    # Paths
    data_p = "../data/base/cross_validation/train_CV0_size10000.csv"
    model_path = "../trained_models/base/airl_CV0_size10000.pt"  # Adjust as necessary
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"

    # Analyze the dataset to get df_analysis for generating trajectories
    df_analysis, df_multiple_routes, df_single_route, df_per_path = analyze_dataset(data_p)

    # Load model and environment
    from network_env import RoadWorld
    from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
    from model.policy import PolicyCNN
    from model.value import ValueCNN
    from model.discriminator import DiscriminatorAIRLCNN

    # Initialize environment
    od_list, od_dist = ini_od_dist(data_p)
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
    def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
        gamma = 0.95  # Adjust gamma as needed
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

        policy_net.eval()
        value_net.eval()
        discrim_net.eval()

        return policy_net, value_net, discrim_net

    policy_net, value_net, discriminator_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Generate trajectories matching the 'path_count' per path
    generated_trajs = generate_trajectories_per_path(df_per_path, policy_net, env, device, max_steps=50)

    # Compute rewards for the generated trajectories
    generated_trajs_with_rewards = compute_rewards_for_generated_trajs(
        generated_trajs, env, discriminator_net, device
    )

    # Identify divergence and convergence points and their rewards
    generated_trajs_with_rewards = identify_divergence_and_convergence_points(generated_trajs_with_rewards)

    # Save generated trajectories with rewards
    output_filename = "./eva/generated_trajectories_with_rewards.csv"
    save_generated_trajectories_matched(generated_trajs_with_rewards, output_filename)

if __name__ == "__main__":
    main()