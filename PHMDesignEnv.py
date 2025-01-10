import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from sklearn import preprocessing

MAX_STEPS = 10000


save_path = r'C:\Users\psl\Desktop\data.npy'
final = np.load(save_path, allow_pickle=True).tolist()
final = pd.DataFrame(data=np.array(final), columns=['month', 'degradation-1-1', 'degradation-1-2', 'degradation-2-1','degradation-2-2','degradation-3-1','degradation-3-2','degradation-4-1','degradation-4-2'])
class PHMDesignEnv(gym.Env):
    """继承自gym的自定义环境类"""
    metadata = {'render.modes': ['human']}

    def __init__(self, final):
        super(PHMDesignEnv, self).__init__()
        self.fore_step = 20
        self.data = final
        self.action_space = spaces.Discrete(9)
        self.maintenance_df = pd.DataFrame([], columns=['month', 'degradation-1-1', 'degradation-1-2', 'degradation-2-1','degradation-2-2','degradation-3-1','degradation-3-2','degradation-4-1','degradation-4-2'])
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 20), dtype=np.float16)
        self.reward = 0
        self.sys_reliability = 1
        self.reward_list = []
        self.sys_r = []
        self.sys_r_cycle = []
        self.sys_r_cycle_total = []



    def reset(self):
        self.current_step = self.fore_step + 1
        self.current_step_1 = self.fore_step + 1
        self.current_step_2 = self.fore_step + 1
        self.current_step_3 = self.fore_step + 1
        self.current_step_4 = self.fore_step + 1
        self.current_step_5 = self.fore_step + 1
        self.current_step_6 = self.fore_step + 1
        self.current_step_7 = self.fore_step + 1
        self.current_step_8 = self.fore_step + 1

        return self._next_observation()


    def _next_observation(self):
    # 获取（读入）基础的数据
        frame = np.array([
            self.data['degradation-1-1'][self.current_step_1 - self.fore_step: self.current_step_1],
            self.data['degradation-1-2'][self.current_step_2 - self.fore_step: self.current_step_2],
            self.data['degradation-2-1'][self.current_step_3 - self.fore_step: self.current_step_3],
            self.data['degradation-2-2'][self.current_step_4 - self.fore_step: self.current_step_4],
            self.data['degradation-3-1'][self.current_step_5 - self.fore_step: self.current_step_5],
            self.data['degradation-3-2'][self.current_step_6 - self.fore_step: self.current_step_6],
            self.data['degradation-4-1'][self.current_step_7 - self.fore_step: self.current_step_7],
            self.data['degradation-4-2'][self.current_step_8 - self.fore_step: self.current_step_8],
        ])
        return frame

    def step(self, action):

        # self.current_month = self.data['month'][self.current_step]


        # if self.current_step > 25000:  # 已经走到头了，重置为0
        if self.current_step > len(self.data['month']) - 2:

            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})
            self.sys_r_cycle_total.append(self.sys_r_cycle)
            self.sys_r_cycle = []
            self.reset()


        elif action == 0:  # 维修后重置
            self.current_step += 1
            self.current_step_1 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 1:
            self.current_step += 1
            self.current_step_2 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 2:
            self.current_step += 1
            self.current_step_3 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 3:
            self.current_step += 1
            self.current_step_4 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 4:
            self.current_step += 1
            self.current_step_5 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 5:
            self.current_step += 1
            self.current_step_6 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 6:
            self.current_step += 1
            self.current_step_7 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 7:
            self.current_step += 1
            self.current_step_8 = self.fore_step + 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})

        elif action == 8:
            self.current_step += 1  # 当前动作数量+1
            self.current_step_1 += 1
            self.current_step_2 += 1
            self.current_step_3 += 1
            self.current_step_4 += 1
            self.current_step_5 += 1
            self.current_step_6 += 1
            self.current_step_7 += 1
            self.current_step_8 += 1
            self.sys_r_cycle.append({'current_step':self.current_step, 'reward': self.reward,'system_reliability':self.sys_reliability,'action':action})
        else:
            pass

        self._take_action(action)
        # delay_modifier = (self.current_step / MAX_STEPS)
        self.sys_reliability = ((self.data['degradation-1-1'][self.current_step_1] + (1-self.data['degradation-1-1'][self.current_step_1])*self.data['degradation-1-2'][self.current_step_2]) * 30000 + (1 - (self.data['degradation-1-1'][self.current_step_1] + (1-self.data['degradation-1-1'][self.current_step_1])*self.data['degradation-1-2'][self.current_step_2])) * 12400 + (self.data['degradation-2-1'][self.current_step_3] + (1 - self.data['degradation-2-1'][self.current_step_3])*self.data['degradation-2-2'][self.current_step_4]) * 30000 + (1 - (self.data['degradation-2-1'][self.current_step_3] + (1 - self.data['degradation-2-1'][self.current_step_3])*self.data['degradation-2-2'][self.current_step_4]))  * 12480 + (self.data['degradation-3-1'][self.current_step_5] + (1 - self.data['degradation-3-1'][self.current_step_5])*self.data['degradation-3-2'][self.current_step_6]) * 30000 + (1 - (self.data['degradation-3-1'][self.current_step_5] + (1 - self.data['degradation-3-1'][self.current_step_5])*self.data['degradation-3-2'][self.current_step_6])) * 21550 + (self.data['degradation-4-1'][self.current_step_7] + (1 - self.data['degradation-4-1'][self.current_step_7])*self.data['degradation-4-2'][self.current_step_8]) * 30000 + (1 - (self.data['degradation-4-1'][self.current_step_7] + (1 - self.data['degradation-4-1'][self.current_step_7])*self.data['degradation-4-2'][self.current_step_8])) * 24960)/120000
        self.compressor = self.data['degradation-1-1'][self.current_step_1]
        self.sys_r.append(self.sys_reliability)

        self.reward = self.reward
        self.reward_list.append(self.reward)


        obs = self._next_observation()  # 下一个状态

        return obs, self.reward, None, {}

    def _take_action(self, action):
      # 提取当前步下的设备状态

        if action ==0:
            self.reward = -10
        if action ==1:
            self.reward = -10
        if action ==2:
            self.reward = -10
        if action ==3:
            self.reward = -10
        if action ==4:
            self.reward = -10
        if action ==5:
            self.reward = -10
        if action ==6:
            self.reward = -10
        if action ==7:
            self.reward = -10

        if action == 8 :
            if self.data['degradation-1-1'][self.current_step_1] < 0.8:
                self.reward = -40
            if self.data['degradation-1-2'][self.current_step_2] < 0.8:
                self.reward = -40
            if self.data['degradation-2-1'][self.current_step_3] < 0.8:
                self.reward = -40
            if self.data['degradation-2-2'][self.current_step_4] < 0.8:
                self.reward = -40
            if self.data['degradation-3-1'][self.current_step_5] < 0.8:
                self.reward = -40
            if self.data['degradation-3-2'][self.current_step_6] < 0.8:
                self.reward = -40
            if self.data['degradation-4-1'][self.current_step_7] < 0.8:
                self.reward = -40
            if self.data['degradation-4-2'][self.current_step_8] < 0.8:
                self.reward = -40
            if self.data['degradation-1-1'][self.current_step_1] + (1-self.data['degradation-1-1'][self.current_step_1])*self.data['degradation-1-2'][self.current_step_2] < 0.9:
                self.reward = -80
            if self.data['degradation-2-1'][self.current_step_3] + (1 - self.data['degradation-2-1'][self.current_step_3])*self.data['degradation-2-2'][self.current_step_4] < 0.92:
                self.reward = -80
            if self.data['degradation-3-1'][self.current_step_5] + (1 - self.data['degradation-3-1'][self.current_step_5])*self.data['degradation-3-2'][self.current_step_6] < 0.91:
                self.reward = -80
            if self.data['degradation-4-1'][self.current_step_7] + (1 - self.data['degradation-4-1'][self.current_step_7])*self.data['degradation-4-2'][self.current_step_8] < 0.90:
                self.reward = -80
            else:
                self.reward = 15
            pass



    def render(self, mode='human', close=False):
        print('{}:{} ,{}:{}'.format('month', self.data['month'][self.current_step],
                                    'reward', self.reward))
