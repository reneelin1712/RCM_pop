import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import csv
from captum.attr import IntegratedGradients, Saliency
import matplotlib.pyplot as plt
import os
import pandas as pd



smoothie = SmoothingFunction().method1
device = torch.device("cpu")


def create_od_set(test_trajs):
    test_od_dict = {}
    for i in range(len(test_trajs)):
        if (test_trajs[i][0], test_trajs[i][-1]) in test_od_dict.keys():
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])].append(i)
        else:
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])] = [i]
    return test_od_dict


def evaluate_edit_dist(test_trajs, learner_trajs, test_od_dict):
    edit_dist_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            min_edit_dist = 1.0
            for test in test_od_trajs:
                edit_dist = editdistance.eval(test, learner) / len(test)
                min_edit_dist = edit_dist if edit_dist < min_edit_dist else min_edit_dist
            edit_dist_list.append(min_edit_dist)
    return np.mean(edit_dist_list)


def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
    bleu_score_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        # get unique reference
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            # print(test_od_trajs)
            # print(learner)
            bleu_score = sentence_bleu(test_od_trajs, learner, smoothing_function=smoothie)
            bleu_score_list.append(bleu_score)
    return np.mean(bleu_score_list)


def evaluate_dataset_dist(test_trajs, learner_trajs):
    test_trajs_str = ['_'.join(traj) for traj in test_trajs]
    # print('test trajs str len', len(test_trajs_str))
    test_trajs_set = set(test_trajs_str)
    # print('test trajs set len', len(test_trajs_set))
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)


def evaluate_log_prob(test_traj, model):
    log_prob_list = []
    for episode in test_traj:
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        log_prob = 0
        for x in episode:
            with torch.no_grad():
                next_prob = torch.log(model.get_action_prob(torch.LongTensor([x.cur_state]).to(device), des)).squeeze()
            next_prob_np = next_prob.detach().cpu().numpy()
            log_prob += next_prob_np[x.action]
        log_prob_list.append(log_prob)
    print(np.mean(log_prob_list))
    return np.mean(log_prob_list)


def evaluate_train_edit_dist(train_traj, learner_traj):
    """This function is used to keep the training epoch with the best edit distance performance on the training data"""
    test_od_dict = create_od_set(train_traj)
    edit_dist = evaluate_edit_dist(train_traj, learner_traj, test_od_dict)
    return edit_dist


def evaluate_metrics(test_traj, learner_traj):
    test_od_dict = create_od_set(test_traj)
    edit_dist = evaluate_edit_dist(test_traj, learner_traj, test_od_dict)
    bleu_score = evaluate_bleu_score(test_traj, learner_traj, test_od_dict)
    js_dist = evaluate_dataset_dist(test_traj, learner_traj)
    print('edit dist', edit_dist)
    print('bleu score', bleu_score)
    print('js distance', js_dist)
    return edit_dist, bleu_score, js_dist


# def evaluate_model(target_od, target_traj, model, env, n_link=714):
#     state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
#     target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
#     learner_traj = []
#     """compute transition matrix for the first OD pair"""
#     curr_ori, curr_des = target_o[0], target_d[0]
#     des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
#     action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
#     state_action = env.state_action[:-1]
#     action_prob[state_action == env.pad_idx] = 0.0
#     transit_prob = np.zeros((n_link, n_link))
#     from_st, ac = np.where(state_action != env.pad_idx)
#     to_st = state_action[state_action != env.pad_idx]
#     transit_prob[from_st, to_st] = action_prob[from_st, ac]
#     """compute sample path for the first OD pair"""
#     sample_path = [str(curr_ori)]
#     curr_state = curr_ori
#     for _ in range(50):
#         if curr_state == curr_des: break
#         next_state = np.argmax(transit_prob[curr_state])
#         sample_path.append(str(next_state))
#         curr_state = next_state
#     learner_traj.append(sample_path)
#     for ori, des in zip(target_o[1:], target_d[1:]):
#         if des == curr_des:
#             if ori == curr_ori:
#                 learner_traj.append(sample_path)
#                 continue
#             else:
#                 curr_ori = ori
#         else:
#             curr_ori, curr_des = ori, des
#             des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
#             action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
#             state_action = env.state_action[:-1]
#             action_prob[state_action == env.pad_idx] = 0.0
#             transit_prob = np.zeros((n_link, n_link))
#             from_st, ac = np.where(state_action != env.pad_idx)
#             to_st = state_action[state_action != env.pad_idx]
#             transit_prob[from_st, to_st] = action_prob[from_st, ac]
#         sample_path = [str(curr_ori)]
#         curr_state = curr_ori
#         for _ in range(50):
#             if curr_state == curr_des: break
#             next_state = np.argmax(transit_prob[curr_state])
#             sample_path.append(str(next_state))
#             curr_state = next_state
#         learner_traj.append(sample_path)
#     evaluate_metrics(target_traj, learner_traj)

def aggregate_attributions(attributions_all):
    total_attr = None
    count = 0

    for attr_traj in attributions_all:
        for attr_step in attr_traj:
            if total_attr is None:
                total_attr = attr_step.copy()
            else:
                total_attr += attr_step
            count += 1

    avg_attr = total_attr / count if count > 0 else total_attr
    return avg_attr

def visualize_attribution(attr, title):
    attr_sum = np.sum(attr, axis=0)  # Sum over channels
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_sum, cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.show()


def compute_expert_attributions(test_trajs, model, env):
    expert_attributions_ig_all = []
    expert_attributions_saliency_all = []

    for traj in test_trajs:
        states = [int(s) for s in traj[:-1]]  # All states except the last one
        destination = int(traj[-1])
        actions = []  # You need to provide the expert actions here

        attributions_ig_traj = []
        attributions_saliency_traj = []

        for idx, state in enumerate(states):
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

            # Get the expert action at this step
            action = actions[idx]  # You need to fill in the expert action

            # Get CNN input
            input_data = model.process_features(state_tensor, des_tensor)
            input_data.requires_grad = True

            # Define forward function
            def forward_func(input_data):
                x = input_data
                x = model.forward(x)
                x_mask = model.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs

            predicted_action = torch.tensor([action], dtype=torch.long).to(device)

            # Compute attributions
            ig = IntegratedGradients(forward_func)
            attributions_ig = ig.attribute(input_data, target=predicted_action)

            saliency = Saliency(forward_func)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)

            # Store attributions
            attributions_ig_traj.append(attributions_ig.squeeze().cpu().detach().numpy())
            attributions_saliency_traj.append(attributions_saliency.squeeze().cpu().detach().numpy())

        expert_attributions_ig_all.append(attributions_ig_traj)
        expert_attributions_saliency_all.append(attributions_saliency_traj)

    return expert_attributions_ig_all, expert_attributions_saliency_all



def evaluate_model(target_od, target_traj, model, env, n_link=437):
    output_dir = "attributions"  # Name of the new folder
    os.makedirs(output_dir, exist_ok=True)

    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
    learner_traj = []
    attributions_ig_all = []
    attributions_saliency_all = []

    """Compute transition matrix and generate trajectory for each OD pair"""
    for traj_idx, (ori, des) in enumerate(zip(target_o, target_d)):
        # Reset environment for the current origin-destination pair
        curr_ori, curr_des = env.reset(st=ori, des=des)

        # Initialize the path and attributions for the current OD pair
        sample_path = [str(curr_ori)]
        curr_state = curr_ori
        attributions_ig_traj = []
        attributions_saliency_traj = []

        for _ in range(50):  # Limit the trajectory to 50 steps
            state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)

            # Get action probabilities from the model for the current state and destination
            with torch.no_grad():
                action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
                action_probs_np = action_probs.cpu().numpy()

            # Choose the most likely action based on the model
            action = np.argmax(action_probs_np)

            # Take a step in the environment
            next_state, reward, done = env.step(action)

            # Append the next state to the path
            if next_state != 437:
                sample_path.append(str(next_state))
            curr_state = next_state

            # Compute attributions using Captum
            # Get CNN input
            input_data = model.process_features(state_tensor, des_tensor)
            input_data.requires_grad = True

            # Define the forward function for Captum
            def forward_func(input_data):
                x = input_data
                x = model.forward(x)
                x_mask = model.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs

            predicted_action = torch.tensor([action], dtype=torch.long).to(device)

            # Compute attributions
            ig = IntegratedGradients(forward_func)
            attributions_ig = ig.attribute(input_data, target=predicted_action)

            saliency = Saliency(forward_func)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)

            # Store attributions for this step
            attributions_ig_traj.append(attributions_ig.squeeze().cpu().detach().numpy())
            attributions_saliency_traj.append(attributions_saliency.squeeze().cpu().detach().numpy())

            if done:
                break

        # Append the generated path and attributions for the current OD pair
        learner_traj.append(sample_path)
        attributions_ig_all.append(attributions_ig_traj)
        attributions_saliency_all.append(attributions_saliency_traj)

    # Evaluate metrics for the generated and target trajectories
    evaluate_metrics(target_traj, learner_traj)

    # Ensure that target_traj and learner_traj have the same length
    assert len(target_traj) == len(learner_traj), "Mismatch in number of trajectories"

    # Save trajectory data and attributions to files
    trajectory_data = []
    for idx, (test_path, learner_path, attr_ig_traj, attr_saliency_traj) in enumerate(
        zip(target_traj, learner_traj, attributions_ig_all, attributions_saliency_all)
    ):
        test_traj_str = "_".join(test_path)
        learner_traj_str = "_".join(learner_path)
        trajectory_data.append([test_traj_str, learner_traj_str])

        # Save attributions for this trajectory to numpy files in the new folder
        np.save(os.path.join(output_dir, f"attributions_ig_traj_{idx}.npy"), attr_ig_traj)
        np.save(os.path.join(output_dir, f"attributions_saliency_traj_{idx}.npy"), attr_saliency_traj)

        # Save attributions to CSV files for Excel
        # Flatten the attributions for each step and create a DataFrame
        attr_ig_flat = [attr_step.flatten() for attr_step in attr_ig_traj]
        attr_saliency_flat = [attr_step.flatten() for attr_step in attr_saliency_traj]

        # Create DataFrames
        df_attr_ig = pd.DataFrame(attr_ig_flat)
        df_attr_saliency = pd.DataFrame(attr_saliency_flat)

        # Save to CSV files
        df_attr_ig.to_csv(os.path.join(output_dir, f"attributions_ig_traj_{idx}.csv"), index=False)
        df_attr_saliency.to_csv(os.path.join(output_dir, f"attributions_saliency_traj_{idx}.csv"), index=False)

        # Save to Excel files
        df_attr_ig.to_excel(os.path.join(output_dir, f"attributions_ig_traj_{idx}.xlsx"), index=False)
        df_attr_saliency.to_excel(os.path.join(output_dir, f"attributions_saliency_traj_{idx}.xlsx"), index=False)


    # Save trajectory data to CSV
    with open("trajectory_with_timestep.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Test Trajectory", "Learner Trajectory"])
        csv_writer.writerows(trajectory_data)

    # Aggregate attributions across all trajectories
    avg_attr_ig = aggregate_attributions(attributions_ig_all)
    avg_attr_saliency = aggregate_attributions(attributions_saliency_all)

    # Visualize the aggregated attributions
    visualize_attribution(avg_attr_ig, "Average Integrated Gradients Attribution - Model")
    visualize_attribution(avg_attr_saliency, "Average Saliency Attribution - Model")

    # If you have expert attributions, you can compute and compare them
    # For demonstration purposes, we'll assume you have actions for the expert trajectories
    # expert_attributions_ig_all, expert_attributions_saliency_all = compute_expert_attributions(
    #     target_traj, model, env
    # )

    # # If expert attributions are available, aggregate and visualize them
    # avg_attr_ig_expert = aggregate_attributions(expert_attributions_ig_all)
    # avg_attr_saliency_expert = aggregate_attributions(expert_attributions_saliency_all)

    # # Visualize expert attributions
    # visualize_attribution(avg_attr_ig_expert, "Average Integrated Gradients Attribution - Expert")
    # visualize_attribution(avg_attr_saliency_expert, "Average Saliency Attribution - Expert")

    # # Compare model and expert attributions
    # diff_attr_ig = avg_attr_ig - avg_attr_ig_expert
    # diff_attr_saliency = avg_attr_saliency - avg_attr_saliency_expert

    # # Visualize the difference in attributions
    # visualize_attribution(diff_attr_ig, "Difference in Integrated Gradients Attribution (Model - Expert)")
    # visualize_attribution(diff_attr_saliency, "Difference in Saliency Attribution (Model - Expert)")