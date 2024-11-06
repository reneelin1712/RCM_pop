import numpy as np
import math
import time
import torch
import torch.nn.functional as F
import wandb
from torch import nn
import os  # For checking file existence
import csv

from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from network_env import RoadWorld
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.torch import to_device
from utils.evaluation import evaluate_model, evaluate_log_prob, evaluate_train_edit_dist
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, minmax_normalization, load_train_sample, load_test_traj

torch.backends.cudnn.enabled = False

def update_params_airl(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).long().to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).long().to(device)
    bad_masks = torch.from_numpy(np.stack(batch.bad_mask)).long().to(device)
    actions = torch.from_numpy(np.stack(batch.action)).long().to(device)
    destinations = torch.from_numpy(np.stack(batch.destination)).long().to(device)
    next_states = torch.from_numpy(np.stack(batch.next_state)).long().to(device)

    with torch.no_grad():
        values = value_net(states, destinations)
        next_values = value_net(next_states, destinations)
        fixed_log_probs = policy_net.get_log_prob(states, destinations, actions)

    # Retrieve hyperparameters from wandb.config
    gamma = wandb.config.gamma
    tau = wandb.config.tau
    l2_reg = wandb.config.l2_reg
    clip_epsilon = wandb.config.clip_epsilon
    max_grad_norm = wandb.config.max_grad_norm
    optim_epochs = wandb.config.optim_epochs
    optim_batch_size = wandb.config.optim_batch_size
    epoch_disc = wandb.config.epoch_disc

    """Update discriminator"""
    for _ in range(epoch_disc):
        # Randomly select a batch from expert_traj
        indices = torch.from_numpy(
            np.random.choice(
                expert_st.shape[0],
                min(states.shape[0], expert_st.shape[0]),
                replace=False
            )
        ).long()
        s_expert_st = expert_st[indices].to(device)
        s_expert_des = expert_des[indices].to(device)
        s_expert_ac = expert_ac[indices].to(device)
        s_expert_next_st = expert_next_st[indices].to(device)

        with torch.no_grad():
            expert_log_probs = policy_net.get_log_prob(s_expert_st, s_expert_des, s_expert_ac)

        g_o = discrim_net(states, destinations, actions, fixed_log_probs, next_states)
        e_o = discrim_net(s_expert_st, s_expert_des, s_expert_ac, expert_log_probs, s_expert_next_st)
        loss_pi = -F.logsigmoid(-g_o).mean()
        loss_exp = -F.logsigmoid(e_o).mean()
        discrim_loss = loss_pi + loss_exp
        optimizer_discrim.zero_grad()
        discrim_loss.backward()
        optimizer_discrim.step()

    """Get advantage estimation from the trajectories"""
    rewards = discrim_net.calculate_reward(states, destinations, actions, fixed_log_probs, next_states).squeeze()
    advantages, returns = estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device)

    """Perform mini-batch PPO update"""
    value_loss, policy_loss = 0, 0
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)
        states_shuffled = states[perm].clone()
        destinations_shuffled = destinations[perm].clone()
        actions_shuffled = actions[perm].clone()
        returns_shuffled = returns[perm].clone()
        advantages_shuffled = advantages[perm].clone()
        fixed_log_probs_shuffled = fixed_log_probs[perm].clone()
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b = states_shuffled[ind]
            destinations_b = destinations_shuffled[ind]
            actions_b = actions_shuffled[ind]
            advantages_b = advantages_shuffled[ind]
            returns_b = returns_shuffled[ind]
            fixed_log_probs_b = fixed_log_probs_shuffled[ind]
            batch_value_loss, batch_policy_loss = ppo_step(
                policy_net, value_net, optimizer_policy, optimizer_value, 1,
                states_b, destinations_b, actions_b, returns_b,
                advantages_b, fixed_log_probs_b, clip_epsilon, l2_reg,
                max_grad_norm
            )
            value_loss += batch_value_loss.item()
            policy_loss += batch_policy_loss.item()
    return discrim_loss.item(), value_loss, policy_loss

def save_model(model_path):
    policy_statedict = policy_net.state_dict()
    value_statedict = value_net.state_dict()
    discrim_statedict = discrim_net.state_dict()
    outdict = {"Policy": policy_statedict,
               "Value": value_statedict,
               "Discrim": discrim_statedict}
    torch.save(outdict, model_path)
    # Log the model artifact to W&B
    artifact = wandb.Artifact('airl-model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

def load_model(model_path):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discriminator Model loaded Successfully")

def main_loop():
    global best_edit, start_time, log_interval, max_iter_num

    best_edit = 1.0

    for i_iter in range(1, max_iter_num + 1):
        """Generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        discrim_net.to_device(torch.device('cpu'))
        batch, _ = agent.collect_samples(min_batch_size, mean_action=False)
        discrim_net.to(device)
        discrim_net.to_device(device)

        discrim_loss, value_loss, policy_loss = update_params_airl(batch, i_iter)
        if i_iter % log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {i_iter}/{max_iter_num} | Elapsed Time: {elapsed_time:.2f}s")
            print(f"Discriminator Loss: {discrim_loss:.4f} | Value Loss: {value_loss:.4f} | Policy Loss: {policy_loss:.4f}")

            learner_trajs = agent.collect_routes_with_OD(test_od, mean_action=True)
            edit_dist = evaluate_train_edit_dist(test_trajs, learner_trajs)
            print(f"Edit Distance: {edit_dist:.4f} | Best Edit Distance: {best_edit:.4f}")

            if edit_dist < best_edit:
                best_edit = edit_dist
                save_model(model_p)
                print("Model saved.")

            print("---")

            # Log metrics to W&B
            wandb.log({
                'Iteration': i_iter,
                'Elapsed Time': elapsed_time,
                'Discriminator Loss': discrim_loss,
                'Value Loss': value_loss,
                'Policy Loss': policy_loss,
                'Edit Distance': edit_dist,
                'Best Edit Distance': best_edit,
                # Log hyperparameters
                'learning_rate': wandb.config.learning_rate,
                'gamma': wandb.config.gamma,
                'tau': wandb.config.tau,
                'l2_reg': wandb.config.l2_reg,
                'clip_epsilon': wandb.config.clip_epsilon,
                'optim_epochs': wandb.config.optim_epochs,
                'optim_batch_size': wandb.config.optim_batch_size,
                'seed': wandb.config.seed,
                # Add other hyperparameters if needed
            })

def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def train():
    wandb.init(project='RCM-pop', entity='reneelin2024')
    config = wandb.config

    global policy_net, value_net, discrim_net
    global optimizer_policy, optimizer_value, optimizer_discrim
    global env, device, expert_st, expert_des, expert_ac, expert_next_st
    global agent, test_od, test_trajs, model_p
    global max_iter_num, log_interval, start_time, best_edit

    # Hyperparameters
    log_std = -0.0  # This might stay constant
    gamma = config.gamma
    tau = config.tau
    l2_reg = config.l2_reg
    learning_rate = config.learning_rate
    clip_epsilon = config.clip_epsilon
    num_threads = config.num_threads
    min_batch_size = config.min_batch_size
    eval_batch_size = config.eval_batch_size
    log_interval = config.log_interval
    save_mode_interval = config.save_mode_interval
    max_grad_norm = config.max_grad_norm
    seed = config.seed
    epoch_disc = config.epoch_disc
    optim_epochs = config.optim_epochs
    optim_batch_size = config.optim_batch_size
    cv = config.cv
    size = config.size
    max_iter_num = config.max_iter_num
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    train_p = "../data/base/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/base/cross_validation/test_CV%d.csv" % cv

    # Generate a unique model path with hyperparameters
    def sanitize(value):
        return str(value).replace('.', '_')

    hyperparam_str = f"lr{sanitize(learning_rate)}_bs{optim_batch_size}_gamma{sanitize(gamma)}_tau{sanitize(tau)}_clip{sanitize(clip_epsilon)}_epoch{optim_epochs}"
    run_id = wandb.run.id
    model_p = f"../trained_models/base/airl_{hyperparam_str}_run{run_id}.pt"

    # Initialize road environment
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))
    # Load features
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

    # Seeding
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define actor and critic
    policy_net = PolicyCNN(
        env.n_actions, env.policy_mask, env.state_action,
        path_feature_pad, edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        env.pad_idx
    ).to(device)
    value_net = ValueCNN(
        path_feature_pad, edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]
    ).to(device)
    discrim_net = DiscriminatorAIRLCNN(
        env.n_actions, gamma, env.policy_mask,
        env.state_action, path_feature_pad, edge_feature_pad,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
        path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
        env.pad_idx
    ).to(device)
    policy_net.to_device(device)
    value_net.to_device(device)
    discrim_net.to_device(device)

    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=learning_rate)
    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=learning_rate)

    # Load expert trajectory
    expert_st, expert_des, expert_ac, expert_next_st = env.import_demonstrations(train_p)
    to_device(device, expert_st, expert_des, expert_ac, expert_next_st)
    print('Done loading expert data... number of episodes: %d' % len(expert_st))

    # Load test data
    test_trajs, test_od = load_train_sample(train_p)

    # Create agent
    agent = Agent(env, policy_net, device, custom_reward=None, num_threads=num_threads)
    print('Agent constructed...')

    # Start training
    start_time = time.time()
    best_edit = 1.0
    main_loop()
    print('Training time', time.time() - start_time)

    # Evaluate model
    if os.path.exists(model_p):
        load_model(model_p)
        test_trajs, test_od = load_test_traj(test_p)
        start_time = time.time()
        evaluate_model(test_od, test_trajs, policy_net, env)
        print('Test time:', time.time() - start_time)
        # Evaluate log probability
        test_trajs = env.import_demonstrations_step(test_p)
        evaluate_log_prob(test_trajs, policy_net)
    else:
        print(f"Model file {model_p} does not exist. Skipping model evaluation.")

    # Finish W&B run
    wandb.finish()

if __name__ == '__main__':
    # Define the sweep configuration
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'metric': {'goal': 'minimize', 'name': 'Best Edit Distance'},
        'parameters': {
            'learning_rate': {'distribution': 'uniform', 'min': 1e-4, 'max': 1e-3},
            'gamma': {'values': [0.95, 0.99]},
            'tau': {'values': [0.9, 0.95]},
            'l2_reg': {'values': [1e-3, 1e-4]},
            'clip_epsilon': {'values': [0.1, 0.2, 0.3]},
            'num_threads': {'values': [4]},
            'min_batch_size': {'values': [4096, 8192]},
            'eval_batch_size': {'values': [4096, 8192]},
            'log_interval': {'values': [10, 20]},
            'save_mode_interval': {'values': [50]},
            'max_grad_norm': {'values': [5, 10]},
            'seed': {'values': [1, 42, 100]},
            'epoch_disc': {'values': [1, 2]},
            'optim_epochs': {'values': [10, 20]},
            'optim_batch_size': {'values': [64, 128]},
            'cv': {'values': [0]},
            'size': {'values': [10000]},
            'max_iter_num': {'values': [1000, 2000]},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project='RCM-pop')

    # Run the sweep agent
    wandb.agent(sweep_id, function=train, count=10)  # Adjust 'count' as needed
