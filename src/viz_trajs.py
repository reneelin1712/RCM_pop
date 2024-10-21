import os
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

    model_dict = torch.load(model_path, map_location=device, weights_only=True)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    policy_net.eval()  # Set to evaluation mode
    value_net.eval()
    discrim_net.eval()

    return policy_net, value_net, discrim_net


def prepare_input_data(env, test_trajs):
    example_traj = test_trajs[0]
    states = [int(s) for s in example_traj[:-1]]  # All states except the last one
    destination = int(example_traj[-1])  # The destination is the last state
    return states, destination


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
    policy_net, value_net, discrim_net = load_model(model_path, device, env, path_feature_pad, edge_feature_pad)

    # Load test trajectories
    test_trajs, test_od = load_test_traj(test_p)

    total_attr_ig = None
    total_attr_saliency = None
    count = 0

    for traj_idx, traj in enumerate(test_trajs):
        states_list = [int(s) for s in traj[:-1]]
        destination = int(traj[-1])

        for state in states_list:
            input_data = get_cnn_input(policy_net, state, destination, device)
            input_data.requires_grad = True

            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

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

            ig = IntegratedGradients(forward_func)
            attributions_ig = ig.attribute(input_data, target=predicted_action)

            saliency = Saliency(forward_func)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)

            attr_ig_np = attributions_ig.squeeze().cpu().detach().numpy()
            attr_saliency_np = attributions_saliency.squeeze().cpu().detach().numpy()

            if total_attr_ig is None:
                total_attr_ig = attr_ig_np
                total_attr_saliency = attr_saliency_np
            else:
                total_attr_ig += attr_ig_np
                total_attr_saliency += attr_saliency_np

            count += 1

    avg_attr_ig = total_attr_ig / count
    avg_attr_saliency = total_attr_saliency / count

    avg_attr_ig_sum = np.sum(avg_attr_ig, axis=0)
    avg_attr_saliency_sum = np.sum(avg_attr_saliency, axis=0)

    # Save averaged Integrated Gradients attribution image
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_ig_sum, cmap='hot')
    plt.title('Averaged Integrated Gradients Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'averaged_ig_attribution.png'))
    plt.close()

    # Save averaged Saliency attribution image
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_attr_saliency_sum, cmap='hot')
    plt.title('Averaged Saliency Attribution')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'averaged_saliency_attribution.png'))
    plt.close()

    num_channels = avg_attr_ig.shape[0]
    for channel_idx in range(num_channels):
        channel_attr_ig = avg_attr_ig[channel_idx]
        channel_attr_saliency = avg_attr_saliency[channel_idx]

        # Save channel-wise Integrated Gradients attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_ig, cmap='hot')
        plt.title(f'Averaged IG Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'averaged_ig_attribution_feature_{channel_idx}.png'))
        plt.close()

        # Save channel-wise Saliency attribution
        plt.figure(figsize=(6, 4))
        plt.imshow(channel_attr_saliency, cmap='hot')
        plt.title(f'Averaged Saliency Attribution - Feature {channel_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'averaged_saliency_attribution_feature_{channel_idx}.png'))
        plt.close()


interpret_model()



# import numpy as np
# import torch
# from network_env import RoadWorld
# from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
# from model.policy import PolicyCNN
# from model.value import ValueCNN
# from model.discriminator import DiscriminatorAIRLCNN
# from core.agent import Agent
# from utils.evaluation import evaluate_model
# import pandas as pd
# import matplotlib.pyplot as plt
# # Compute attributions
# from captum.attr import IntegratedGradients, Saliency


# def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
#     gamma = 0.99  # discount factor
#     policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
#                            path_feature_pad, edge_feature_pad,
#                            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
#                            env.pad_idx).to(device)
#     value_net = ValueCNN(path_feature_pad, edge_feature_pad,
#                          path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
#     discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
#                                        env.state_action, path_feature_pad, edge_feature_pad,
#                                        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
#                                        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
#                                        env.pad_idx).to(device)

#     model_dict = torch.load(model_path, map_location=device,weights_only=True)
#     policy_net.load_state_dict(model_dict['Policy'])
#     value_net.load_state_dict(model_dict['Value'])
#     discrim_net.load_state_dict(model_dict['Discrim'])

#     policy_net.eval()  # Set to evaluation mode
#     value_net.eval()
#     discrim_net.eval()

#     return policy_net, value_net, discrim_net


# def prepare_input_data(env, test_trajs):
#     # For simplicity, select the first trajectory from the test set
#     example_traj = test_trajs[0]
#     # Get the states and destination from the trajectory
#     states = [int(s) for s in example_traj[:-1]]  # All states except the last one
#     destination = int(example_traj[-1])  # The destination is the last state
#     return states, destination


# def get_cnn_input(policy_net, state, des,device):
#     state = torch.tensor([state], dtype=torch.long).to(device)
#     des = torch.tensor([des], dtype=torch.long).to(device)
#     # Process features to get the CNN input
#     input_data = policy_net.process_features(state, des)
#     return input_data

# def interpret_model():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#     # Initialize lists to store aggregated attributions
#     total_attr_ig = None
#     total_attr_saliency = None
#     count = 0

#     # Loop over multiple trajectories
#     for traj_idx, traj in enumerate(test_trajs):
#         states_list = [int(s) for s in traj[:-1]]  # All states except the last one
#         print('states_list',states_list)
#         destination = int(traj[-1])  # The destination is the last state

#         for state in states_list:
#             # Get CNN input
#             input_data = get_cnn_input(policy_net, state, destination, device)
#             input_data.requires_grad = True

#             state_tensor = torch.tensor([state], dtype=torch.long).to(device)
#             des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

#             # Define the forward function
#             def forward_func(input_data):
#                 x = input_data
#                 x = policy_net.forward(x)
#                 x_mask = policy_net.policy_mask[state_tensor]
#                 x = x.masked_fill((1 - x_mask).bool(), -1e32)
#                 action_probs = torch.softmax(x, dim=1)
#                 return action_probs

#             with torch.no_grad():
#                 output = forward_func(input_data)
#             predicted_action = torch.argmax(output, dim=1)

#             # Integrated Gradients
#             ig = IntegratedGradients(forward_func)
#             attributions_ig = ig.attribute(input_data, target=predicted_action)

#             # Saliency
#             saliency = Saliency(forward_func)
#             attributions_saliency = saliency.attribute(input_data, target=predicted_action)

#             # Convert to numpy arrays
#             attr_ig_np = attributions_ig.squeeze().cpu().detach().numpy()
#             attr_saliency_np = attributions_saliency.squeeze().cpu().detach().numpy()

#             # Aggregate attributions
#             if total_attr_ig is None:
#                 total_attr_ig = attr_ig_np
#                 total_attr_saliency = attr_saliency_np
#             else:
#                 total_attr_ig += attr_ig_np
#                 total_attr_saliency += attr_saliency_np

#             count += 1

#     # Compute average attributions
#     avg_attr_ig = total_attr_ig / count
#     avg_attr_saliency = total_attr_saliency / count

#     # Sum over the channels to get a single 2D map
#     avg_attr_ig_sum = np.sum(avg_attr_ig, axis=0)
#     avg_attr_saliency_sum = np.sum(avg_attr_saliency, axis=0)

#     # Plot the averaged Integrated Gradients attribution
#     plt.figure(figsize=(6, 4))
#     plt.imshow(avg_attr_ig_sum, cmap='hot')
#     plt.title('Averaged Integrated Gradients Attribution')
#     plt.colorbar()
#     plt.show()

#     # Plot the averaged Saliency attribution
#     plt.figure(figsize=(6, 4))
#     plt.imshow(avg_attr_saliency_sum, cmap='hot')
#     plt.title('Averaged Saliency Attribution')
#     plt.colorbar()
#     plt.show()

#     num_channels = avg_attr_ig.shape[0]
#     for channel_idx in range(num_channels):
#         channel_attr_ig = avg_attr_ig[channel_idx]
#         channel_attr_saliency = avg_attr_saliency[channel_idx]

#         # Plot the channel-wise Integrated Gradients attribution
#         plt.figure(figsize=(6, 4))
#         plt.imshow(channel_attr_ig, cmap='hot')
#         plt.title(f'Averaged IG Attribution - Feature {channel_idx}')
#         plt.colorbar()
#         plt.show()

#         # Plot the channel-wise Saliency attribution
#         plt.figure(figsize=(6, 4))
#         plt.imshow(channel_attr_saliency, cmap='hot')
#         plt.title(f'Averaged Saliency Attribution - Feature {channel_idx}')
#         plt.colorbar()
#         plt.show()



# interpret_model()