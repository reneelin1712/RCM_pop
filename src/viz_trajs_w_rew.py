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

    # Ensure output directories exist
    output_dir = 'output_img/attribution'
    os.makedirs(output_dir, exist_ok=True)

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

    total_attr_ig_policy = None
    total_attr_saliency_policy = None
    total_attr_ig_discrim = None
    total_attr_saliency_discrim = None
    count = 0

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

            saliency = Saliency(forward_func_policy)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)

            attr_ig_np = attributions_ig.squeeze().cpu().detach().numpy()
            attr_saliency_np = attributions_saliency.squeeze().cpu().detach().numpy()

            if total_attr_ig_policy is None:
                total_attr_ig_policy = attr_ig_np
                total_attr_saliency_policy = attr_saliency_np
            else:
                total_attr_ig_policy += attr_ig_np
                total_attr_saliency_policy += attr_saliency_np

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

            saliency_disc = Saliency(forward_func_discriminator)
            attributions_saliency_disc = saliency_disc.attribute(input_data_disc, target=None)

            attr_ig_disc_np = attributions_ig_disc.squeeze().cpu().detach().numpy()
            attr_saliency_disc_np = attributions_saliency_disc.squeeze().cpu().detach().numpy()

            if total_attr_ig_discrim is None:
                total_attr_ig_discrim = attr_ig_disc_np
                total_attr_saliency_discrim = attr_saliency_disc_np
            else:
                total_attr_ig_discrim += attr_ig_disc_np
                total_attr_saliency_discrim += attr_saliency_disc_np

            count += 1

    # Compute average attributions for policy network
    avg_attr_ig_policy = total_attr_ig_policy / count
    avg_attr_saliency_policy = total_attr_saliency_policy / count

    # Compute average attributions for discriminator network
    avg_attr_ig_discrim = total_attr_ig_discrim / count
    avg_attr_saliency_discrim = total_attr_saliency_discrim / count

    # Sum over the channels to get a single 2D map for policy network
    avg_attr_ig_policy_sum = np.sum(avg_attr_ig_policy, axis=0)
    avg_attr_saliency_policy_sum = np.sum(avg_attr_saliency_policy, axis=0)

    # Sum over the channels to get a single 2D map for discriminator network
    avg_attr_ig_discrim_sum = np.sum(avg_attr_ig_discrim, axis=0)
    avg_attr_saliency_discrim_sum = np.sum(avg_attr_saliency_discrim, axis=0)

    # Save averaged Integrated Gradients attribution image for policy network
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_ig_policy_sum, cmap='hot')
    plt.title('Policy Network - Averaged Integrated Gradients Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_averaged_ig_attribution.png'))
    plt.close()

    # Save averaged Saliency attribution image for policy network
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_saliency_policy_sum, cmap='hot')
    plt.title('Policy Network - Averaged Saliency Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'policy_averaged_saliency_attribution.png'))
    plt.close()

    # Save averaged Integrated Gradients attribution image for discriminator network
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_ig_discrim_sum, cmap='hot')
    plt.title('Discriminator Network - Averaged Integrated Gradients Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_averaged_ig_attribution.png'))
    plt.close()

    # Save averaged Saliency attribution image for discriminator network
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_saliency_discrim_sum, cmap='hot')
    plt.title('Discriminator Network - Averaged Saliency Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_averaged_saliency_attribution.png'))
    plt.close()

    # Save channel-wise attributions for policy network
    num_channels_policy = avg_attr_ig_policy.shape[0]
    for channel_idx in range(num_channels_policy):
        channel_attr_ig = avg_attr_ig_policy[channel_idx]
        channel_attr_saliency = avg_attr_saliency_policy[channel_idx]

        # Save channel-wise Integrated Gradients attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_ig, cmap='hot')
        plt.title(f'Policy Network - Averaged IG Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'policy_averaged_ig_attribution_feature_{channel_idx}.png'))
        plt.close()

        # Save channel-wise Saliency attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_saliency, cmap='hot')
        plt.title(f'Policy Network - Averaged Saliency Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'policy_averaged_saliency_attribution_feature_{channel_idx}.png'))
        plt.close()

    # Save channel-wise attributions for discriminator network
    num_channels_discrim = avg_attr_ig_discrim.shape[0]
    for channel_idx in range(num_channels_discrim):
        channel_attr_ig = avg_attr_ig_discrim[channel_idx]
        channel_attr_saliency = avg_attr_saliency_discrim[channel_idx]

        # Save channel-wise Integrated Gradients attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_ig, cmap='hot')
        plt.title(f'Discriminator Network - Averaged IG Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'discriminator_averaged_ig_attribution_feature_{channel_idx}.png'))
        plt.close()

        # Save channel-wise Saliency attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_saliency, cmap='hot')
        plt.title(f'Discriminator Network - Averaged Saliency Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'discriminator_averaged_saliency_attribution_feature_{channel_idx}.png'))
        plt.close()

    print("Interpretation complete. Averaged attributions saved in the 'output_img/attribution' directory.")


if __name__ == "__main__":
    interpret_model()
