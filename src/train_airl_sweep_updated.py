import numpy as np
import math
import time
import torch
import torch.nn.functional as F
import wandb
from torch import nn
import os  # To check for file existence

from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from network_env import RoadWorld
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.torch import to_device
from utils.evaluation import (
    evaluate_model,
    evaluate_log_prob,
    evaluate_train_edit_dist,
    evaluate_diversity_metrics_single
)
from utils.load_data import (
    ini_od_dist,
    load_path_feature,
    load_link_feature,
    minmax_normalization,
    load_train_sample,
    load_test_traj,
)

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
    epoch_disc = wandb.config.epoch_disc
    optim_epochs = wandb.config.optim_epochs
    optim_batch_size = wandb.config.optim_batch_size
    clip_epsilon = wandb.config.clip_epsilon
    l2_reg = wandb.config.l2_reg
    max_grad_norm = wandb.config.max_grad_norm

    for _ in range(epoch_disc):
        indices = torch.from_numpy(
            np.random.choice(
                expert_st.shape[0],
                min(states.shape[0], expert_st.shape[0]),
                replace=False,
            )
        ).long()
        s_expert_st = expert_st[indices].to(device)
        s_expert_des = expert_des[indices].to(device)
        s_expert_ac = expert_ac[indices].to(device)
        s_expert_next_st = expert_next_st[indices].to(device)

        with torch.no_grad():
            expert_log_probs = policy_net.get_log_prob(
                s_expert_st, s_expert_des, s_expert_ac
            )
        g_o = discrim_net(
            states, destinations, actions, fixed_log_probs, next_states
        )
        e_o = discrim_net(
            s_expert_st,
            s_expert_des,
            s_expert_ac,
            expert_log_probs,
            s_expert_next_st,
        )
        loss_pi = -F.logsigmoid(-g_o).mean()
        loss_exp = -F.logsigmoid(e_o).mean()
        discrim_loss = loss_pi + loss_exp
        optimizer_discrim.zero_grad()
        discrim_loss.backward()
        optimizer_discrim.step()

    rewards = discrim_net.calculate_reward(
        states, destinations, actions, fixed_log_probs, next_states
    ).squeeze()
    advantages, returns = estimate_advantages(
        rewards,
        masks,
        bad_masks,
        values,
        next_values,
        gamma,
        tau,
        device,
    )

    value_loss, policy_loss = 0, 0
    optim_iter_num = int(
        math.ceil(states.shape[0] / optim_batch_size)
    )
    for _ in range(optim_epochs):
        perm = torch.randperm(states.shape[0]).to(device)
        states, destinations, actions, returns, advantages, fixed_log_probs = (
            states[perm],
            destinations[perm],
            actions[perm],
            returns[perm],
            advantages[perm],
            fixed_log_probs[perm],
        )
        for i in range(optim_iter_num):
            ind = slice(
                i * optim_batch_size,
                min(
                    (i + 1) * optim_batch_size, states.shape[0]
                ),
            )
            (
                states_b,
                destinations_b,
                actions_b,
                advantages_b,
                returns_b,
                fixed_log_probs_b,
            ) = (
                states[ind],
                destinations[ind],
                actions[ind],
                advantages[ind],
                returns[ind],
                fixed_log_probs[ind],
            )
            (
                batch_value_loss,
                batch_policy_loss,
            ) = ppo_step(
                policy_net,
                value_net,
                optimizer_policy,
                optimizer_value,
                1,
                states_b,
                destinations_b,
                actions_b,
                returns_b,
                advantages_b,
                fixed_log_probs_b,
                clip_epsilon,
                l2_reg,
                max_grad_norm,
            )
            value_loss += batch_value_loss.item()
            policy_loss += batch_policy_loss.item()
    return discrim_loss.item(), value_loss, policy_loss

def save_model(model_path):
    # Before saving, check for NaNs or Infs in parameters
    invalid_params = False
    for name, param in policy_net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Invalid values in policy_net parameter {name}")
            invalid_params = True
            break
    for name, param in value_net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Invalid values in value_net parameter {name}")
            invalid_params = True
            break
    for name, param in discrim_net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Invalid values in discrim_net parameter {name}")
            invalid_params = True
            break
    if invalid_params:
        print("Model parameters contain invalid values. Skipping model save.")
        return

    # Proceed to save model
    policy_statedict = policy_net.state_dict()
    value_statedict = value_net.state_dict()
    discrim_statedict = discrim_net.state_dict()
    outdict = {
        "Policy": policy_statedict,
        "Value": value_statedict,
        "Discrim": discrim_statedict,
    }
    torch.save(outdict, model_path)
    artifact = wandb.Artifact(
        f'airl-model-{wandb.run.name}',
        type='model',
        metadata=dict(wandb.config),
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

def load_model(model_path):
    try:
        model_dict = torch.load(model_path)
        policy_net.load_state_dict(model_dict['Policy'])
        print("Policy Model loaded Successfully")
        value_net.load_state_dict(model_dict['Value'])
        print("Value Model loaded Successfully")
        discrim_net.load_state_dict(model_dict['Discrim'])
        print("Discriminator Model loaded Successfully")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Cannot load the model.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

def train():
    wandb.init(project='RCM-pop-diverse', entity='reneelin2024',resume='allow')
    config = wandb.config

    global policy_net, value_net, discrim_net
    global optimizer_policy, optimizer_value, optimizer_discrim
    global env, device, expert_st, expert_des, expert_ac, expert_next_st
    global agent, test_od, test_trajs, model_p

    try:
        # Set device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize environment and data
        # Load paths
        edge_p = "../data/base/edge.txt"
        network_p = "../data/base/transit.npy"
        path_feature_p = "../data/base/feature_od.npy"
        cv = config.cv
        size = config.size
        train_p = "../data/base/cross_validation/train_CV%d_size%d.csv" % (cv, size)
        test_p = "../data/base/cross_validation/test_CV%d.csv" % cv

        # Generate a unique model path with hyperparameters
        def sanitize(value):
            return str(value).replace('.', '_')

        # Optionally, include more hyperparameters as needed
        hyperparam_str = f"lr{sanitize(config.learning_rate)}_bs{config.optim_batch_size}_gamma{sanitize(config.gamma)}_tau{sanitize(config.tau)}_clip{sanitize(config.clip_epsilon)}_epoch{config.optim_epochs}"
        run_id = wandb.run.id  # Alternatively, use wandb.run.name
        model_p = f"../trained_models/base/airl_{hyperparam_str}_run{run_id}.pt"

        # Initialize road environment
        od_list, od_dist = ini_od_dist(train_p)
        env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))

        # Load path-level and link-level features
        path_feature, path_max, path_min = load_path_feature(path_feature_p)
        edge_feature, link_max, link_min = load_link_feature(edge_p)
        path_feature = minmax_normalization(path_feature, path_max, path_min)
        path_feature_pad = np.zeros(
            (env.n_states, env.n_states, path_feature.shape[2])
        )
        path_feature_pad[
            : path_feature.shape[0], : path_feature.shape[1], :
        ] = path_feature
        edge_feature = minmax_normalization(edge_feature, link_max, link_min)
        edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
        edge_feature_pad[: edge_feature.shape[0], :] = edge_feature

        # Seeding
        seed = config.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Define actor and critic
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
        discrim_net = DiscriminatorAIRLCNN(
            env.n_actions,
            config.gamma,
            env.policy_mask,
            env.state_action,
            path_feature_pad,
            edge_feature_pad,
            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
            path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
            env.pad_idx,
        ).to(device)

        policy_net.to_device(device)
        value_net.to_device(device)
        discrim_net.to_device(device)

        optimizer_policy = torch.optim.Adam(
            policy_net.parameters(), lr=config.learning_rate
        )
        optimizer_value = torch.optim.Adam(
            value_net.parameters(), lr=config.learning_rate
        )
        optimizer_discrim = torch.optim.Adam(
            discrim_net.parameters(), lr=config.learning_rate
        )

        # Load expert trajectory
        expert_st, expert_des, expert_ac, expert_next_st = env.import_demonstrations(
            train_p
        )
        to_device(device, expert_st, expert_des, expert_ac, expert_next_st)
        print('Done loading expert data... number of episodes: %d' % len(expert_st))

        # Load test data
        test_trajs, test_od = load_train_sample(train_p)

        # Create agent
        agent = Agent(
            env,
            policy_net,
            device,
            custom_reward=None,
            num_threads=config.num_threads,
        )
        print('Agent constructed...')

        start_time = time.time()
        best_edit = float('inf')  # Ensure best_edit is initialized appropriately

        for i_iter in range(1, config.max_iter_num + 1):
            try:
                batch, _ = agent.collect_samples(
                    config.min_batch_size, mean_action=False
                )
            except ValueError as e:
                # Specific exception for NaNs detected in action probabilities
                if "NaNs detected in action_prob during action selection." in str(e):
                    print(f"NaNs detected during action selection at iteration {i_iter}: {e}")
                    # Log the problematic hyperparameters
                    wandb.log({'exception': str(e),
                               'learning_rate': config.learning_rate,
                               'hyperparameters': dict(config)}, commit=False)
                    # Save the hyperparameters to a local file
                    with open('problematic_hyperparams.txt', 'a') as f:
                        f.write(f"NaNs encountered with hyperparameters: {dict(config)}\n")
                    # Save the model before exiting
                    save_model(model_p)
                    break  # Exit training loop and proceed to the next hyperparameter combination
                else:
                    # For other ValueErrors, re-raise the exception
                    raise
            except Exception as e:
                print(f"An error occurred during agent.collect_samples at iteration {i_iter}: {e}")
                wandb.log({'exception': str(e),
                           'learning_rate': config.learning_rate,
                           'hyperparameters': dict(config)}, commit=False)
                # Save the model before exiting
                save_model(model_p)
                break  # Exit training loop and proceed to the next hyperparameter combination

            try:
                discrim_loss, value_loss, policy_loss = update_params_airl(batch, i_iter)
            except Exception as e:
                print(f"An error occurred during update_params_airl at iteration {i_iter}: {e}")
                wandb.log({'exception': str(e),
                           'learning_rate': config.learning_rate,
                           'hyperparameters': dict(config)}, commit=False)
                # Save the model before exiting
                save_model(model_p)
                break  # Exit training loop

            if i_iter % config.log_interval == 0:
                elapsed_time = time.time() - start_time
                try:
                    learner_trajs = agent.collect_routes_with_OD(
                        test_od, mean_action=True
                    )
                    edit_dist = evaluate_train_edit_dist(test_trajs, learner_trajs)

                    # # Compute conditional entropy
                    # learner_diversity_results = evaluate_diversity_metrics_single(learner_trajs, label='Learner')
                    # #conditional_entropy = learner_diversity_results['Average Conditional Entropy']
                    # transition_entropy = learner_diversity_results['Average Transition Entropy']

                    # Compute diversity metrics including n-gram transition entropy
                    n = 3  # Set n to the desired n-gram size
                    learner_diversity_results = evaluate_diversity_metrics_single(learner_trajs, label='Learner', n=n)
                    ngram_transition_entropy = learner_diversity_results[f'Average {n}-gram Transition Entropy']


                    # Compute the composite score
                    lambda_entropy = 0.5  # Adjust this weighting factor as needed
                    # score = edit_dist - lambda_entropy * transition_entropy
                    score = edit_dist - lambda_entropy * ngram_transition_entropy

                except Exception as e:
                    print(f"An error occurred during evaluation at iteration {i_iter}: {e}")
                    wandb.log({'exception': str(e),
                            'learning_rate': config.learning_rate,
                            'hyperparameters': dict(config)}, commit=False)
                    # Save the model before exiting
                    save_model(model_p)
                    break  # Exit training loop

                if edit_dist < best_edit:
                    best_edit = edit_dist
                    wandb.run.summary['Best Edit Distance'] = best_edit
                    save_model(model_p)
                    print(f"Model saved to {model_p}")

                # Log metrics to WandB
                wandb.log(
                    {
                        'Iteration': i_iter,
                        'Elapsed Time': elapsed_time,
                        'Discriminator Loss': discrim_loss,
                        'Value Loss': value_loss,
                        'Policy Loss': policy_loss,
                        'Edit Distance': edit_dist,
                        # 'Transition Entropy': transition_entropy,
                        f'{n}-gram Transition Entropy': ngram_transition_entropy,
                        'Score': score,
                        'Best Edit Distance': best_edit,
                        'learning_rate': config.learning_rate,
                        # Log other hyperparameters if needed
                    }
                )

                # Print metrics to terminal
                print(f"Iteration {i_iter} - Edit Distance: {edit_dist}, Conditional Entropy ngram: {ngram_transition_entropy}, Score: {score}")


        # After training loop, save the model regardless
        save_model(model_p)
        print(f"Model saved to {model_p} at the end of training.")

        # Evaluate model
        if os.path.exists(model_p):
            load_model(model_p)
            test_trajs, test_od = load_test_traj(test_p)
            start_time = time.time()
            evaluate_model('test_dataset',test_od, test_trajs, policy_net, env)
            print('Test time:', time.time() - start_time)

            # Evaluate log probability
            test_trajs = env.import_demonstrations_step(test_p)
            evaluate_log_prob(test_trajs, policy_net)
        else:
            print(f"Model file {model_p} does not exist. Skipping model evaluation.")

    except Exception as e:
        # Log the exception and proceed to the next hyperparameter combination
        print(f"An error occurred during training: {e}")
        # Save the model before exiting
        save_model(model_p)
        wandb.log({'exception': str(e),
                   'learning_rate': config.learning_rate,
                   'hyperparameters': dict(config)}, commit=False)
        # Save the hyperparameters to a local file
        with open('problematic_hyperparams.txt', 'a') as f:
            f.write(f"Exception during training with hyperparameters: {dict(config)}\n")
    finally:
        # Finish W&B run
        wandb.finish()

if __name__ == '__main__':
    # Sweep configuration with more hyperparameters
    sweep_config = {
        'method': 'bayes', #random
        # 'metric': {'goal': 'minimize', 'name': 'Best Edit Distance'},
        'metric': {'goal': 'minimize', 'name': 'Score'},
        'parameters': {
           'learning_rate': {'distribution': 'uniform', 'min': 2e-4, 'max': 4e-4},
            'optim_batch_size': {'values': [32, 64, 128]},
            'max_iter_num': {'values': [1000]},
            'log_interval': {'values': [10, 20, 50]},
            'optim_epochs': {'values': [10, 20, 30]},
            'gamma': {'values': [0.95, 0.99]},
            'tau': {'values': [0.9, 0.95]},
            'clip_epsilon': {'values': [0.1, 0.2, 0.3]},
            'min_batch_size': {'values': [4096, 8192]},
            'max_grad_norm': {'values': [5, 10]},
            'epoch_disc': {'values': [1, 2]},
            'l2_reg': {'values': [1e-3, 1e-4]},
            'seed': {'values': [1, 42, 100]},
            'cv': {'values': [0]},
            'size': {'values': [10000]},
            'num_threads': {'values': [4]},
        },
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project='RCM-pop-diverse')

    # Run the sweep agent
    wandb.agent(sweep_id, function=train, count=5)


