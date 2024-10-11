import json
import numpy as np
import torch
from flask import Flask, jsonify
from flask_cors import CORS  # Import the CORS extension
from network_env import RoadWorld
from utils.load_data import load_test_traj, ini_od_dist, load_path_feature, load_link_feature, minmax_normalization
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from core.agent import Agent
from utils.evaluation import evaluate_model
import pandas as pd
import random

app = Flask(__name__)

# Enable CORS for all routes and all domains
CORS(app) # specify origins=["http://localhost:5173"]

with open('edges.geojson') as f:
    edges_data = json.load(f)
# Print the structure of the loaded data to verify it
print(edges_data)

# Dummy model function that returns a hardcoded path
def get_dummy_path():
    path = "405_172_356_362_193_185_177_404_194_173_143_167_407_157_159_169"
    return path

# Load model function from `eva.py`
def load_model(model_path, device, env, path_feature_pad, edge_feature_pad):
    gamma = 0.99
    edge_data = pd.read_csv('../data/base/edge.txt')

    # Instantiate models
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

    # Load the model state dicts
    model_dict = torch.load(model_path, map_location=device)
    policy_net.load_state_dict(model_dict['Policy'])
    value_net.load_state_dict(model_dict['Value'])
    discrim_net.load_state_dict(model_dict['Discrim'])

    return policy_net, value_net, discrim_net

# Initialize model and environment once during server startup
device = torch.device('cpu')

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

def generate_trajectory():
    n_link = 437  # Ensure n_link is consistent with the evaluate_model function
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    # Select a random index from the available test OD pairs
    random_idx = random.randint(0, len(test_od) - 1)
    curr_ori, curr_des = test_od[random_idx, 0], test_od[random_idx, 1]
    des_ts = (torch.ones_like(state_ts) * curr_des).to(device)

    # Get action probabilities from the model
    action_prob = policy_net.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # Shape should be (n_link, n_actions)

    # Trim the state_action array as done in evaluate_model (to avoid mismatch in dimensions)
    state_action = env.state_action[:-1]  # Shape should be (n_link - 1)

    # Mask invalid actions (actions corresponding to pad_idx)
    action_prob[state_action == env.pad_idx] = 0.0

    # Initialize the transition probability matrix
    transit_prob = np.zeros((n_link, n_link))  # Matrix to store transition probabilities

    # Populate the transition matrix with valid probabilities
    from_st, ac = np.where(state_action != env.pad_idx)
    to_st = state_action[state_action != env.pad_idx]
    transit_prob[from_st, to_st] = action_prob[from_st, ac]

    # Generate the sample path for the first OD pair
    sample_path = [str(curr_ori)]
    curr_state = curr_ori

    for _ in range(50):  # Limit the trajectory to 50 steps
        if curr_state == curr_des:
            break
        next_state = np.argmax(transit_prob[curr_state])
        sample_path.append(str(next_state))
        curr_state = next_state

    print('sample_path',sample_path)
    # Convert sample_path to the required format "405_172_356_..."
    return '_'.join(sample_path)

def generate_trajectory_for_od(curr_ori, curr_des):
    n_link = 437  # Ensure n_link is consistent with the evaluate_model function
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    
    # Reset environment with the selected origin and destination
    env.reset(st=curr_ori, des=curr_des)
    
    des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
    
    # Get action probabilities from the model
    action_prob = policy_net.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # Shape should be (n_link, n_actions)

    # Trim the state_action array as done in evaluate_model (to avoid mismatch in dimensions)
    state_action = env.state_action[:-1]  # Shape should be (n_link - 1)

    # Mask invalid actions (actions corresponding to pad_idx)
    action_prob[state_action == env.pad_idx] = 0.0

    # Generate the sample path for the selected OD pair
    sample_path = [str(curr_ori)]
    curr_state = curr_ori

    for _ in range(50):  # Limit the trajectory to 50 steps
        if curr_state == curr_des:
            break
        
        # Select the action with the highest probability
        action = np.argmax(action_prob[curr_state])  # Choose the most likely action
        
        # Take a step in the environment using the selected action
        next_state, reward, done = env.step(action)

        # Append the next state to the path
        if next_state !=437:
            sample_path.append(str(next_state))
        curr_state = next_state
        
        # If done or if the next state is invalid (dead-end), terminate the path generation
        if done or next_state == env.pad_idx:
            print(f"Dead-end reached at state {curr_state}. Ending trajectory generation.")
            break

        # # Add the next state to the path
        # sample_path.append(str(next_state))
        # curr_state = next_state

    # Convert the sample path to the required format "405_172_356_..."
    return '_'.join(sample_path)  # Return the path as a string




def get_route(path):
    # Split the path string by '_'
    edge_names = path.split('_')
    
    # Prepare a list to hold the GeoJSON features
    features = []

    # Fetch geometries for each edge in the path and convert to GeoJSON features
    for feature in edges_data['features']:  # Loop through all features in edges_data
        edge_name = feature['properties']['name']
        if edge_name in edge_names:
            # Append the feature directly as it is in the edges.geojson
            features.append(feature)

    # Return the route as a GeoJSON FeatureCollection
    return {
        "type": "FeatureCollection",
        "features": features
    }


# Function to get coordinates for origin and destination from edges.geojson
def get_edge_coordinates(edge_name):
    for feature in edges_data['features']:
        if feature['properties']['name'] == edge_name:
            return feature['geometry']['coordinates']
    return None  # If the edge is not found

# GET endpoint to return a generated path in GeoJSON format
# @app.route('/get_route', methods=['GET'])
# def get_route_endpoint():
#     # Generate the path using the model
#     path = generate_trajectory()
    
#     # Convert the path into the corresponding GeoJSON features
#     route = get_route(path)
    
#     if len(route) == 0:
#         return jsonify({"error": "No matching edges found"}), 404
    
#     # Return the route in GeoJSON-like format
#     return jsonify(route)
@app.route('/get_route', methods=['GET'])
def get_route_endpoint():
    # Select a random index from the available test OD pairs
    random_idx = random.randint(0, len(test_od) - 1)
    curr_ori, curr_des = test_od[random_idx, 0], test_od[random_idx, 1]

    # Convert the random test trajectory into the corresponding GeoJSON features
    test_traj = '_'.join(map(str, test_trajs[random_idx]))  # Convert the test trajectory to the desired format
    test_route = get_route(test_traj)  # Get the corresponding GeoJSON

    # Generate the trajectory using the model for the same test OD pair
    generated_path = generate_trajectory_for_od(curr_ori, curr_des)  # Use the same OD for comparison
    generated_route = get_route(generated_path)  # Get the corresponding GeoJSON

    # Get origin and destination coordinates from the edges
    origin_coords = get_edge_coordinates(str(curr_ori))
    destination_coords = get_edge_coordinates(str(curr_des))

    # Check if the routes have features
    if len(generated_route['features']) == 0 or len(test_route['features']) == 0:
        return jsonify({"error": "No matching edges found"}), 404

    # Convert `curr_ori` and `curr_des` from `numpy.int64` to standard Python integers
    origin = int(curr_ori)
    destination = int(curr_des)

    # Return both the generated and test trajectories in GeoJSON format and raw string format, along with origin/destination coordinates
    return jsonify({
        "generated_route": generated_route,
        "test_route": test_route,
        "generated_path": generated_path,  # Return the generated trajectory in raw format
        "test_path": test_traj,  # Return the test trajectory in raw format
        "origin": origin,  # Add origin edge name
        "destination": destination,  # Add destination edge name
        "origin_coords": origin_coords,  # Add origin coordinates
        "destination_coords": destination_coords  # Add destination coordinates
    })

# Test endpoint to confirm the server is running
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "The server is running!"})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
