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
from captum.attr import FeatureAblation


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

    # Feature names (as per your earlier code)
    feature_names = [
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
    states, destination = prepare_input_data(env, test_trajs)

    # We'll interpret the first state in the trajectory
    state = states[0]

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
    print('predicted_action',predicted_action)

    # Integrated Gradients
    ig = IntegratedGradients(forward_func)
    attributions_ig = ig.attribute(input_data, target=predicted_action)

    # Saliency
    saliency = Saliency(forward_func)
    attributions_saliency = saliency.attribute(input_data, target=predicted_action)

    # Convert attributions to numpy arrays and remove the batch dimension
    attr_ig = attributions_ig.squeeze().cpu().detach().numpy()
    attr_saliency = attributions_saliency.squeeze().cpu().detach().numpy()

    # Aggregate attributions for each channel
    channel_importance = np.sum(np.abs(attr_ig), axis=(1, 2))
    print('attr_ig.shape:', attr_ig.shape)  # Should be (num_channels, height, width)
    print('Number of channels in attr_ig:', attr_ig.shape[0])
    print('len(feature_names):', len(feature_names))  # Should be 20


    # Rank features
    ranked_indices = np.argsort(-channel_importance)
    sorted_importance = channel_importance[ranked_indices]
    ranked_features = [feature_names[i] for i in ranked_indices]

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': ranked_features,
        'Importance Score': sorted_importance
    })

    print(feature_importance_df)

    # Feature Ablation
    feature_ablation = FeatureAblation(forward_func)
    attributions_ablation = feature_ablation.attribute(input_data, target=predicted_action)
    attr_ablation = attributions_ablation.squeeze().cpu().detach().numpy()

    # Aggregate attributions for each channel (Feature Ablation)
    channel_importance_ablation = np.sum(np.abs(attr_ablation), axis=(1, 2))

    # Rank features (Feature Ablation)
    ranked_indices_ablation = np.argsort(-channel_importance_ablation)
    sorted_importance_ablation = channel_importance_ablation[ranked_indices_ablation]
    ranked_features_ablation = [feature_names[i] for i in ranked_indices_ablation]

    # Create DataFrame (Feature Ablation)
    feature_importance_ablation_df = pd.DataFrame({
        'Feature': ranked_features_ablation,
        'Importance Score': sorted_importance_ablation
    })
    print("Feature Importance from Feature Ablation:")
    print(feature_importance_ablation_df)



    # Visualize Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(ranked_features[::-1], sorted_importance[::-1])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    plt.show()

    # Visualize Feature Importance (Feature Ablation)
    plt.figure(figsize=(10, 6))
    plt.barh(ranked_features_ablation[::-1], sorted_importance_ablation[::-1])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Ranking (Feature Ablation)')
    plt.tight_layout()
    plt.show()

    # Sum over the channels to get a single 2D map
    attr_ig_sum = np.sum(attr_ig, axis=0)
    attr_saliency_sum = np.sum(attr_saliency, axis=0)
    attr_ablation_sum = np.sum(attr_ablation, axis=0)

    # Plot the Integrated Gradients attribution map
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_ig_sum, cmap='hot')
    plt.title('Integrated Gradients Attribution Map')
    plt.colorbar()
    plt.show()

    # Plot the Integrated Gradients attribution
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_ig_sum, cmap='hot')
    plt.title('Integrated Gradients Attribution')
    plt.colorbar()
    plt.show()

    # Plot the Saliency attribution
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_saliency_sum, cmap='hot')
    plt.title('Saliency Attribution')
    plt.colorbar()
    plt.show()

    # Plot the Feature Ablation attribution map
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_ablation_sum, cmap='hot')
    plt.title('Feature Ablation Attribution Map')
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