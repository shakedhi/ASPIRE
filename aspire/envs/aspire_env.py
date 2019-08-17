import gym
from gym import spaces
import pandas as pd
import numpy as np


class AspireEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 detectors,                                                   # Action labels
                 cost_func=lambda r: r if r < 1 else min(1 + np.log2(r), 6),  # Cost function
                 t_func=lambda r: 1,                                          # Reward for TP or TN
                 fp_func=lambda r: -r,                                        # Reward for FP
                 fn_func=lambda r: -r,                                        # Reward for FN
                 illegal_reward=-10000                                        # Cost for illegal action
        ):
        
        # init action space
        self.detectors = detectors
        self.termination_actions_ = ['benign', 'malicious']
        self.scan_actions_ = list(self.detectors.keys())
        self.action_labels = self.scan_actions_ + self.termination_actions_
        self.action_space = spaces.Discrete(n=len(self.action_labels))

        # init state
        self.observation_space = spaces.Box(low=-2, high=1, shape=(len(self.scan_actions_), 1))
        self.term_state_ = np.full(shape=(self.observation_space.shape[0],), fill_value=-2, dtype=np.float32)
        
        # init costs
        self.costs = dict(manalyze=0.75, pefile=0.7, byte3g=3.99, opcode2g=42.99) # mean cost for each detector
        self.cost_func = cost_func

        # init rewards
        self.illegal_reward = illegal_reward
        self.t_func = t_func
        self.fp_func = fp_func
        self.fn_func = fn_func
        
        # reset
        self.current = None
        self.state = np.full(shape=(self.observation_space.shape[0],), fill_value=-1, dtype=np.float32)
        self.scanned = []
        self.reward = 0
        self.done = False
        self.pred = ''
        self.costs = []

    def step(self, action):
        action_label = self.action_labels[action]
        self.scanned.append(action_label)
        
        if action_label in self.scan_actions_:
            if self.scanned.count(action_label) > 1:
                return self.step_illegal_(pred='DUP')            # Duplicate action
            return self.step_scan_(action, action_label)
        
        elif action_label in self.termination_actions_:
            self.done = True
            if len(self.scanned) == 1:
                return self.step_illegal_(pred='DIR')            # Direct classification
            return self.step_term_(action_label)
        
        raise ValueError('action value is outside of action_space')
    
    def step_scan_(self, action, action_label):
        result, costs = self.detectors[action_label].detect(self.current['File'])
        self.costs.append(costs)
        self.state[action] = result           # update state
        self.reward += self.cost_func(costs)  # update reward
        return self.state, 0, self.done, {}
        
    def step_illegal_(self, pred):
        self.state = self.term_state_
        self.reward = self.illegal_reward
        self.pred = pred
        return self.state, self.reward, self.done, {}

    def step_term_(self, action_label):
        if self.current['Label'] == 1:
            # Actual malicious
            if action_label is 'malicious':
                self.reward = self.t_func(self.reward)
                self.pred = 'TP'
            else:
                self.reward = self.fn_func(self.reward)
                self.pred = 'FN'
        else:
            # Actual benign
            if action_label is 'benign':
                self.reward = self.t_func(self.reward)
                self.pred = 'TN'
            else:
                self.reward = self.fp_func(self.reward)
                self.pred = 'FP'
        
        return self.state, self.reward, self.done, {}
    
    def reset(self, file_data=None):
        self.current = file_data
        self.state = np.full(shape=(self.observation_space.shape[0],), fill_value=-1, dtype=np.float32)
        self.scanned = []
        self.reward = 0
        self.done = False
        self.pred = ''
        self.costs = []
        return self.state

    def render(self, mode='human', close=False):
        if self.done:
            label = 'm' if self.current['Label'] == 1 else 'b'
            name = self.current['Name'][:10]
            actions = ','.join(self.scanned)
            print(f'[{label}-{name}] Pred={self.pred:<3} R={self.reward:>8} A={actions}')
