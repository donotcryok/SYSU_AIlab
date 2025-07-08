import os
import random
import copy
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        pass


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        pass


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

                print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        # 关闭TensorBoard写入器
        self.writer.close()
    
    def _update_network(self):
        """更新Q网络"""
        # 从回放缓冲区中采样一批经验
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 提取批次数据
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.array(batch[4])).to(self.device)
        
        # 计算当前Q值和目标Q值
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        self.writer.add_scalar('Train/Loss', loss.item(), self.train_step)
    
    def _adjust_learning_rate(self):
        """学习率衰减"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_min, param_group['lr'] * self.lr_decay)
        
        self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.train_step)
