import datetime
import os
import xlwt  # 用于处理表格文件
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
wb = xlwt.Workbook()
ws = wb.add_sheet("data")
from env1 import Maze
SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def main(args):
    env = Maze()
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    # Build a tensorboard
    rewards = []
    moving_avg_rewards = []
    eps_steps = []
    log_dir = os.path.split(os.path.abspath(__file__))[0] + "/logs/train/" + SEQUENCE
    writer = SummaryWriter(log_dir)
    for i_eps in range(1, args.train_steps):
        s = env.reset()
        done = False
        episode_reward = 0.
        sum_k = 0
        sum_rr = 0
        for i_step in range(args.max_steps):
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, k, rr = env.step(action)
            if done and i_step != args.max_episode_steps:
                dw = True
            else:
                dw = False
            sum_k += k
            sum_rr += rr
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            episode_reward += r
            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, i_eps)
                replay_buffer.count = 0
        rewards.append(episode_reward)
        eps_steps.append(i_eps)
        writer.add_scalars('rewards', {'raw': rewards[-1]}, i_eps)
        print("第", i_eps, "轮的收益为：", episode_reward)
        ws.write(i_eps, 0, i_eps)
        ws.write(i_eps, 1, episode_reward)
        ws.write(i_eps, 2, int(sum_k))
        ws.write(i_eps, 3, int(sum_rr))
    wb.save("step_reward.xls")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument('--max_steps', default=500, type=int)
    parser.add_argument("--train_steps", type=int, default=int(700), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=int(500), help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    main(args)
