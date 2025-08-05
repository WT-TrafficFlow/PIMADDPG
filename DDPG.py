#  建立 DDPG 结构
import tensorflow as tf
import tensorflow.contrib as tc

class DDPG():
    def __init__(self, name, num_state, num_action, num_other_aciton, layer_norm=True):
        state_input = tf.placeholder(shape=[None, num_state], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, num_action], dtype=tf.float32)
        other1_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other2_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other3_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other4_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other5_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other6_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        other7_action_input = tf.placeholder(shape=[None, num_other_aciton], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        def actor_network(name):
            #  为 actor 建立两层的全连接网络
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
                x = state_input
                x = tf.layers.dense(x, 128)
                if layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)     #  进行标准化
                x = tf.nn.relu(x)    #  使用 relu 进行激活
                x = tf.layers.dense(x, 128)
                if layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                #  输出层
                x = tf.layers.dense(x, num_action, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x) # --------------------------将 tanh 改为 sigmoid
            return x

        def critic_network(name, action_input):
            #  critic_network 需 要 输 入 state 以 及 所 有 agent 当 前 的 action 信 息 （ action_input, other_action_input）
            with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
                x = state_input
                x = tf.layers.dense(x, 128)
                if layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.concat([x, action_input], axis=-1)  # axis 为-1 等价于默认值 1：向行方向进行合并（第一个维度保持不变）
                x = tf.layers.dense(x, 128)
                if layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)
                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3))
            return x

        self.action_output = actor_network(name + "_actor")
        self.critic_output = critic_network(name + '_critic', action_input=tf.concat([action_input,
                                                                                      other1_action_input,other2_action_input,other3_action_input,
                                                                                      other4_action_input,other5_action_input,other6_action_input,
                                                                                      other7_action_input], axis=1))
        self.state_input = state_input
        self.action_input = action_input
        self.other1_action_input = other1_action_input
        self.other2_action_input = other2_action_input
        self.other3_action_input = other3_action_input
        self.other4_action_input = other4_action_input
        self.other5_action_input = other5_action_input
        self.other6_action_input = other6_action_input
        self.other7_action_input = other7_action_input

        self.reward = reward
        self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        # actor 训练的目标是最大化 Q 值
        self.actor_loss = -tf.reduce_mean(critic_network(name + '_critic',
                                                         action_input=tf.concat(
                                                             [self.action_output, self.other1_action_input,self.other2_action_input,
                                                              self.other3_action_input,self.other4_action_input,self.other5_action_input,
                                                              self.other6_action_input,self.other7_action_input], axis=1)))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        # critic 训练的目标是最小化 Q 估计值和 Q 实际值之间的差距
        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

    def train_actor(self, state, other1_action,other2_action,other3_action,other4_action,other5_action,other6_action,other7_action, sess):
        sess.run(self.actor_train, {self.state_input: state, self.other1_action_input: other1_action,self.other2_action_input: other2_action,
                                    self.other3_action_input: other3_action,self.other4_action_input: other4_action,
                                    self.other5_action_input: other5_action,self.other6_action_input: other6_action,
                                    self.other7_action_input: other7_action})

    def train_critic(self, state, action, other1_action,other2_action,other3_action, other4_action,other5_action,other6_action,other7_action,target, sess):
        sess.run(self.critic_train, {self.state_input: state, self.action_input: action, self.other1_action_input:
            other1_action, self.other2_action_input:other2_action, self.other3_action_input:other3_action,
                                     self.other4_action_input:other4_action,
                                     self.other5_action_input:other5_action,
                                     self.other6_action_input:other6_action,
                                     self.other7_action_input:other7_action,self.target_Q: target})

    def action(self, state, sess):
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other1_action, other2_action,other3_action,other4_action,other5_action,other6_action,other7_action,sess):
        return sess.run(self.critic_output, {self.state_input: state, self.action_input: action,
                                             self.other1_action_input: other1_action,
                                             self.other2_action_input: other2_action,
                                             self.other3_action_input: other3_action,
                                             self.other4_action_input: other4_action,
                                             self.other5_action_input: other5_action,
                                             self.other6_action_input: other6_action,
                                             self.other7_action_input: other7_action})

#  建立经验回放池以及采样机制
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    #  增加一次经验
    def add(self, obs_t, action, reward, obs_tp1, done):
         # obs_t：当前的状态值；obs_tp1：采取该 action 后得到的下一步的状态值；done：一次迭代是否结束，当前动作，以及采取动作后的奖励
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            #  如果新的经验数量超过 maxsize，则会覆盖之前的旧经验
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        # idxes 为[0, len(self._storage)-1]之间的随机数组成的长度为 batch_size 的 list（将经验打乱）
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        #  按照 idxes 中的 index 顺序将 self._storage 中对应的值分别存入上述的 list 中
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    #  采样一批次的经验
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

#  初始化参数以及参数的更新
def create_init_update(online_name, target_name, tau=0.9999):
   import tensorflow as tf
   online_var = [i for i in tf.trainable_variables() if online_name in i.name]
   target_var = [i for i in tf.trainable_variables() if target_name in i.name]
   target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
   # target 网络使用 soft update 的方法更新参数（相当于把该网络固定住），target 网络参数变化小
   target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var,
                                                                                                   target_var)]
   return target_init, target_update

#  智能体的训练
def  train_agent(agent_ddpg,  agent_ddpg_target,  agent_memory,  agent_actor_target_update,
                    agent_critic_target_update, sess, other1_actor, other2_actor, other3_actor,
                 other4_actor,other5_actor,other6_actor,other7_actor): # --------------------------------------->
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(32) # 挑选32个经验数组
    #  获取当前所有 agent 采取的动作
    act_batch = total_act_batch[:, 0, :]
    other1_act_batch = total_act_batch[:, 1, :]
    other2_act_batch = total_act_batch[:, 2, :]
    other3_act_batch = total_act_batch[:, 3, :]
    other4_act_batch = total_act_batch[:, 4, :]
    other5_act_batch = total_act_batch[:, 5, :]
    other6_act_batch = total_act_batch[:, 6, :]
    other7_act_batch = total_act_batch[:, 7, :]

    #  获取当前的状态
    obs_batch = total_obs_batch[:, 0, :]
    #  获取采用当前动作后的下一个状态
    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other1_obs_batch = total_next_obs_batch[:, 1, :]
    next_other2_obs_batch = total_next_obs_batch[:, 2, :]
    next_other3_obs_batch = total_next_obs_batch[:, 3, :]
    next_other4_obs_batch = total_next_obs_batch[:, 4, :]
    next_other5_obs_batch = total_next_obs_batch[:, 5, :]
    next_other6_obs_batch = total_next_obs_batch[:, 6, :]
    next_other7_obs_batch = total_next_obs_batch[:, 7, :]

    #  获取下一个状态下其他 agent 的动作
    next_other1_action = other1_actor.action(next_other1_obs_batch, sess) # ---------------------------------->
    # print('next_other1_action',next_other1_action)
    next_other2_action = other2_actor.action(next_other2_obs_batch, sess)
    next_other3_action = other3_actor.action(next_other3_obs_batch, sess)
    next_other4_action = other4_actor.action(next_other4_obs_batch, sess)
    next_other5_action = other5_actor.action(next_other5_obs_batch, sess)
    next_other6_action = other6_actor.action(next_other6_obs_batch, sess)
    next_other7_action = other7_actor.action(next_other7_obs_batch, sess)

    # target 是基于下一个状态的
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, # 0.9999

                                                                     action=agent_ddpg.action(next_obs_batch, sess),

                                                                     other1_action=next_other1_action,
                                                                     other2_action=next_other2_action,
                                                                     other3_action=next_other3_action,
                                                                     other4_action=next_other4_action,
                                                                     other5_action=next_other5_action,
                                                                     other6_action=next_other6_action,
                                                                     other7_action=next_other7_action,
                                                                     sess=sess)
    agent_ddpg.train_actor(state=obs_batch, other1_action=other1_act_batch, other2_action=other2_act_batch, other3_action=other3_act_batch,
                           other4_action=other4_act_batch,other5_action=other5_act_batch,other6_action=other6_act_batch,
                           other7_action=other7_act_batch,sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other1_action=other1_act_batch,other2_action=other2_act_batch,other3_action=other3_act_batch,
                            other4_action=other4_act_batch,other5_action=other5_act_batch,other6_action=other6_act_batch,
                            other7_action=other7_act_batch,
                            target=target, sess=sess)
    sess.run([agent_actor_target_update, agent_critic_target_update])
















