# Drone 환경 DDPG Code 설명

Unity Drone 환경을 해결할 수 있는 DDPG Code를 작성해 보자.

## Import
DDPG를 위한 라이브러리들을 import 합니다

```python
import numpy as np
import numpy.random as nr
import random
import tensorflow as tf
import tensorflow.layers as layer
from collections import deque
from mlagents.envs import UnityEnvironment
```
DQN코드와 유사합니다

## Setting

```python
state_size = 6 # or [80, 80 ,3]
agent_size = 1
action_size = 3

load_model = False
train_mode = True

batch_size = 1024
hidden_layer_size = 512
mem_maxlen = 1000000
discount_factor = 0.99
learning_rate = 0.0001
run_episode = 1000000
update_interval = 100
update_target_rate = 0.001
print_interval = 100
save_interval = 500
epsilon_refresh = True
epsilon_refresh_trig = 0.01
epsilon_decay = 0.99995
noise_std = 0.01

noise_option = 'layer_noise' # 'layer_noise' or None
env_name = "../envs/Drone"
logdir = "../Summary/ddpg"
```
- state_size = 6 : 이 네트워크는 6개의 vector 위치 정보를 사용합니다. [80, 80, 3]으로 설정하며 80x80 rgb 이미지를 사용하며 mlp에서 cnn feature extractor가 변경된 구조로 바뀝니다
- agent_size = 1 : 환경에서 지원하는 agent 개수를 지정합니다.
- action_size = 3 : 이 네트워크는 3개의 continuous한 vector를 결정하여 환경을 수행합니다
- load_model = False : 이전에 학습한 모델을 읽어들이지 않습니다. 만약 이전에 학습한 모델을 읽어들이고 싶다면 True 로 설정합니다.
- train_mode = True : 에이전트를 학습시킵니다. False로 하면 학습을 진행하지 않고 에이전트가 환경을 수행하는 모습을 볼수 있습니다.
- batch_size = 1024 : 한번 모델을 학습할때 리플레이 메모리에서 1024쌍의 데이터를 읽어 학습합니다.
- hidden_layer_size = 512 : fully connected layer의 hidden layer 크기를 512로 설정합니다.
- mem_maxlen = 1000000 : 리플레이 메모리의 최대 크기를 설정합니다.
- discount_factor = 0.99 : reward의 discount factor를 0.99로 설정하여 줍니다.
- learning_rate = 0.0001 : Actor/Critic 네트워크의 Adam Optimizer의 learning rate를 설정하여 줍니다.
- run_episode = 1000000 : 총 1000000의 에피소드를 걸쳐 학습합니다.
- update_interval = 100 : 각 에피소드의 최대 legnth를 정해줍니다.
- update_target_rate = 0.001 : soft update 시 update하는 rate를 결정합니다.
- print_interval = 100 : 100번 episode마다 정보 출력합니다.
- save_interval = 500 : 500번 episode마다 모델 저장합니다.
- epsilon_refresh = True : noise epsilon이 어느 정도 낮아지면 초기화합니다.
- epsilon_refresh_trig = 0.01 : noise epsilon 초기화 시, 초기화하는 수치 결정합니다.
- epsilon_decay = 0.99995 : noise epsilon 감소 정도
- noise_std = 0.01 : 'layer_noise' 옵션을 줄 때 각 weight noise의 std를 설정합니다.
- noise_option = 'ou_noise' : rl exploration을 위한 옵션입니다. None, 'ou_noise', 'layer_noise' 옵션 제공합니다.
- env_name = "../envs/Drone" : 어떤 파일에서 환경을 불러올지 파일 주소를 표시합니다.
- logdir = "../Summary/ddpg" : tf summary에 저장되는 위치를 결정합니다.

## Model
```python
class DDPGModel():
    def __init__(self, state_size, action_size, hidden_layer_size, mem_maxlen, save_path, learning_rate, load_model, batch_size, epsilon, epsilon_decay, update_target_rate, epsilon_min=0.01, discount_factor=0.99, noise_option='ou_noise'):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_laeyr_size = hidden_layer_size

        self.noise_option = noise_option
        if noise_option is None:
            self.noiser = None

        elif noise_option == 'ou_noise':
            self.noiser = OU_noise([agent_size, action_size])
        elif noise_option == 'layer_noise':
            self.noiser = None

        self.actor_model = ActorNetwork(
            self.state_size, self.action_size, self.hidden_laeyr_size, learning_rate, "actor_model")
        self.critic_model = CriticNetwork(
            self.state_size, self.action_size, self.hidden_laeyr_size, learning_rate, "critic_model")

        self.actor_target = ActorNetwork(
            self.state_size, self.action_size, self.hidden_laeyr_size, learning_rate, "actor_target")
        self.critic_target = CriticNetwork(
            self.state_size, self.action_size, self.hidden_laeyr_size, learning_rate, "critic_target")
        self.tvar = tf.trainable_variables()
        self.actor_model.trainer(self.critic_model, batch_size)
        self.update_op_holder = UpdateTargetGraph(
            self.tvar, update_target_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.Session = tf.Session()
        self.load_model = load_model

        self.init = tf.global_variables_initializer()
        self.batch_size = batch_size

        self.Session.run(self.init)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.save_path = save_path
        self.Summary, self.Merge = self.make_Summary()

        if self.load_model == True:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            self.Saver.restore(self.Session, ckpt.model_checkpoint_path)

    def get_action(self, state, train_mode=True):
        # print(state)
        if train_mode:
            if self.noise_option == 'ou_noise':
                out = self.Session.run(self.actor_model.action, feed_dict={
                               self.actor_model.observation: state})
                return out + self.noiser.noise()*self.epsilon
            elif self.noise_option == 'layer_noise':
                self.apply_noise(noise_std)
                out = self.Session.run(self.actor_model.action, feed_dict={
                               self.actor_model.observation: state})
                self.reset_vars()
                return out
            else:
                out = self.Session.run(self.actor_model.action, feed_dict={
                               self.actor_model.observation: state})
                return out
        else:
            out = self.Session.run(self.actor_model.action, feed_dict={
                               self.actor_model.observation: state})
            return out

    def apply_noise(self, noise_std=1):
        var_names = [var for var in tf.global_variables() if 'decide' in var.op.name]
        self.old_var = self.Session.run(var_names)
        var_shapes = [i.shape for i in self.old_var]
        new_var = [i+np.random.normal(0,noise_std,size = j) for i,j in zip(self.old_var,var_shapes)]
        # setting new values
        for i,j in zip(var_names,new_var):
            self.Session.run(i.assign(j))
        return 

    def reset_vars(self):
        var_names = [var for var in tf.global_variables() if 'decide' in var.op.name]

        # setting old values
        for i,j in zip(var_names,self.old_var):
            self.Session.run(i.assign(j))
        return

    def append_sample(self, state, action, reward, next_state, done):
        for i in range(agent_size):
            self.memory.append(
                (state[i], action[i], reward[i], next_state[i], done[i]))

    def save_model(self):
        self.Saver.save(self.Session, self.save_path + "\model.ckpt")

    def train_model(self, print_debug=False):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        critic_action_input = self.Session.run(self.actor_target.action, feed_dict={
                                               self.actor_target.observation: next_states})
        target_q_value = self.Session.run(self.critic_target.value, feed_dict={
                                          self.critic_target.observation: next_states, self.critic_target.action: critic_action_input})

        targets = np.zeros([self.batch_size, 1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + \
                    self.discount_factor*target_q_value[i]

        _, loss = self.Session.run([self.critic_model.Update, self.critic_model.loss], feed_dict={
                                   self.critic_model.observation: states, self.critic_model.action: actions, self.critic_model.true_value: targets})

        action_for_train = self.Session.run(self.actor_model.action, feed_dict={
                                            self.actor_model.observation: states})
        _, grad = self.Session.run([self.actor_model.Update, self.actor_model.policy_Grads], feed_dict={
                                   self.actor_model.observation: states, self.critic_model.observation: states, self.critic_model.action: action_for_train})
        # print(grad)
        return loss

    def update_target(self):
        update(self.Session, self.update_op_holder)

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)
        return tf.summary.FileWriter(logdir=logdir, graph=self.Session.graph), tf.summary.merge_all()

    def Write_Summray(self, reward, loss, episode):
        self.Summary.add_summary(self.Session.run(self.Merge, feed_dict={
                                 self.summary_loss: loss, self.summary_reward: reward}), episode)
```
`DDPGModel`은 DDPG Actor-Critic 모델입니다. 
DDPG는 actor model에서 action을 결정하고, critic model에서 해당 action에 대한 평가를 합니다.

>     `get_action` 함수는 현재 state를 기반으로 action을 return합니다. 학습 과정 중일 때는 action에 노이즈를 주는 과정이 추가될 수 있습니다.

>     `apply_noise` 함수와 `reset_vars` 함수는 weight에 노이즈를 주는 'layer_noise' 옵션일 때 사용 되는 함수입니다.

>     `append_sample` 함수는 replay memory에 정보를 저장 하는 함수입니다.

>     `save_model` 함수는 현재 model 상태를 저장하는 함수입니다.

>     `train_model` 함수는 replay memory에 저장되어 있는 정보로 모델을 soft update합니다.

>     `update_target` 함수는 `train_model` 함수 내에서 호출하는 함수로서 target 모델을 soft update하는 함수입니다.

>     `make_Summary` 함수와 `Write_Summary` 함수는 학습하는 과정을 저장하기 위한 함수입니다.

## Agent
`DDPGModel`에서 사용되는 actor-critic agent들을 살펴봅니다.

### Actor Agent
```python
class ActorNetwork:
    def __init__(self, state_size, action_size, hidden_layer_size, learning_rate, name):
        if isinstance(state_size, list) and len(state_size) == 3:
            self.mode = 'visual'
        elif isinstance(state_size, int):
            self.mode = 'vector'
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_layer_size
        self.name = name
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            if self.mode == 'visual':
                self.observation = tf.placeholder(
                    tf.float32, shape=[None, *self.state_size], name="actor_observation")
                self.L1 = layer.conv2d(
                    self.observation, 64, 3, activation=tf.nn.leaky_relu)
                self.L2 = layer.conv2d(
                    self.L1, 64, 3, activation=tf.nn.leaky_relu)
                self.L3 = layer.flatten(self.L2)
                self.action = layer.dense(
                    self.L3, self.action_size, activation=tf.nn.tanh, name='actor_decide')

            elif self.mode == 'vector':
                self.observation = tf.placeholder(
                    tf.float32, shape=[None, self.state_size], name="actor_observation")
                self.L1 = layer.dense(
                    self.observation, self.hidden_size, activation=tf.nn.leaky_relu)
                self.L2 = layer.dense(
                    self.L1, self.hidden_size, activation=tf.nn.leaky_relu)
                self.L3 = layer.dense(
                    self.L2, self.hidden_size, activation=tf.nn.leaky_relu)
                self.action = layer.dense(
                    self.L3, self.action_size, activation=tf.nn.tanh, name='actor_decide')

        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def trainer(self, critic, batch_size):
        action_Grad = tf.gradients(critic.value, critic.action)
        self.policy_Grads = tf.gradients(
            ys=self.action, xs=self.trainable_var, grad_ys=action_Grad)
        for idx, grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads/batch_size

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.Adam = tf.train.AdamOptimizer(self.learning_rate)
            self.Update = self.Adam.apply_gradients(
                zip(self.policy_Grads, self.trainable_var))
```
`ActorNetwork`는 현재 state에서 가장 올바르다고 생각하는 action을 return하는 모델입니다.

>     `trainer` 함수는 critic 모델과 같이 replay memory에서 batch 크기 만큼 gradient 계산을 합니다.

### Critic Agent
```python
class CriticNetwork:
    def __init__(self, state_size, action_size, hidden_layer_size, learning_rate, name):
        if isinstance(state_size, list) and len(state_size) == 3:
            self.mode = 'visual'
        elif isinstance(state_size, int):
            self.mode = 'vector'
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_size = hidden_layer_size
        self.name = name
        with tf.variable_scope(name):
            if self.mode == 'visual':
                self.observation = tf.placeholder(
                    tf.float32, shape=[None, *self.state_size], name="critic_observation")
                self.O1 = layer.flatten(layer.conv2d(
                    self.observation, 64, 3, activation=tf.nn.leaky_relu))
                self.action = tf.placeholder(
                    tf.float32, shape=[None, self.action_size], name="critic_action")
                self.A1 = layer.dense(
                    self.action, self.hidden_layer_size//2, activation=tf.nn.leaky_relu)
                self.L1 = tf.concat([self.O1, self.A1], 1)
                self.L1 = layer.dense(
                    self.L1, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.L2 = layer.dense(
                    self.L1, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.L3 = layer.dense(
                    self.L2, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.value = layer.dense(self.L3, 1, activation=None)
            elif self.mode == 'vector':
                self.observation = tf.placeholder(
                    tf.float32, shape=[None, self.state_size], name="critic_observation")
                self.O1 = layer.dense(
                    self.observation, self.hidden_layer_size//2, activation=tf.nn.leaky_relu)
                self.action = tf.placeholder(
                    tf.float32, shape=[None, self.action_size], name="critic_action")
                self.A1 = layer.dense(
                    self.action, self.hidden_layer_size//2, activation=tf.nn.leaky_relu)
                self.L1 = tf.concat([self.O1, self.A1], 1)
                self.L1 = layer.dense(
                    self.L1, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.L2 = layer.dense(
                    self.L1, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.L3 = layer.dense(
                    self.L2, self.hidden_layer_size, activation=tf.nn.leaky_relu)
                self.value = layer.dense(self.L3, 1, activation=None)

        self.true_value = tf.placeholder(tf.float32, name="true_value")
        self.loss = tf.losses.huber_loss(self.true_value, self.value)
        self.Adam = tf.train.AdamOptimizer(learning_rate)
        self.Update = self.Adam.minimize(self.loss)
        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
```
`CriticNetwork`는 현재 state와 `ActorNetwork`에서 return한 action을 입력받아 expected value를 계산하는 모델입니다.

## Main
```python
env = UnityEnvironment(file_name=env_name)

default_brain = env.brain_names[0]
brain = env.brains[default_brain]

env_info = env.reset(train_mode=train_mode and not Viewtrain)[
    default_brain]

agent = DDPGModel(state_size,
                  action_size,
                  hidden_layer_size,
                  mem_maxlen,
                  save_path,
                  learning_rate,
                  load_model,
                  batch_size,
                  1.0,
                  epsilon_decay=epsilon_decay,
                  update_target_rate=update_target_rate,
                  discount_factor=discount_factor,
                  noise_option=noise_option)

print("Agent shape looks like: \n{}".format(
    np.shape(env_info.visual_observations)))

reward_memory = deque(maxlen=20)
agent_reward = np.zeros(agent_size)
losses = deque(maxlen=20)
frame_count = 0
for episode in range(run_episode):
    if isinstance(state_size, list) and len(state_size) == 3:
        state = env_info.visual_observations
    elif isinstance(state_size, int):
        state = env_info.vector_observations
    for i in range(update_interval):
        frame_count += 1
        action = agent.get_action(state, train_mode)
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        agent_reward += reward
        done = env_info.local_done
        for idx, don in enumerate(done):
            if don:
                reward_memory.append(agent_reward[idx])
                agent_reward[idx] = 0
        if train_mode:
            agent.append_sample(state, action, reward, next_state,done)
        state = next_state

        if train_mode and len(agent.memory) > agent.batch_size * 2:
            loss = agent.train_model()
            losses.append(loss)
            agent.update_target()

    if epsilon_refresh and agent.epsilon < epsilon_refresh_trig:
        agent.epsilon = 0.9

    if episode % print_interval == 0:
        print("episode({}) - reward: {:.2f}     loss: {:.4f}     epsilon: {:.3f}       memory_len:{}".format(
            episode, np.mean(reward_memory), np.mean(losses),agent.epsilon, len(agent.memory)))
        agent.Write_Summray(np.mean(reward_memory),
                            np.mean(losses), episode)

    if episode % save_interval == 0 and episode != 0:
        print("model saved")
        agent.save_model()
```
`main`에서는 유니티 학습 환경을 불러오고 DDPG 모델을 만든 후 episode를 거치면서 학습하는 과정을 거칩니다.
>     env = UnityEnvironment(file_name=env_name)
>     default_brain = env.brain_names[0]
>     brain = env.brains[default_brain]
>     env_info = env.reset(train_mode=train_mode and not Viewtrain)[default_brain]
>학습을 시작하기 전에 유니티 학습 환경을 불러옵니다. 학습 환경을 reset해 줌으로서 새 학습 환경으로 먼저 설정합니다.


>     for i in range(update_interval):
>             frame_count += 1
>             action = agent.get_action(state, train_mode)
>             env_info = env.step(action)[default_brain]
>             next_state = env_info.vector_observations
>             reward = env_info.rewards
>             agent_reward += reward
>             done = env_info.local_done
>             for idx, don in enumerate(done):
>                 if don:
>                     reward_memory.append(agent_reward[idx])
>                     agent_reward[idx] = 0
>             if train_mode:
>                 agent.append_sample(state, action, reward, next_state,done)
>             state = next_state
>각 에피소드마다 `update_interval` 옵션에 따라 환경에서 memory를 쌓습니다.

>     if train_mode and len(agent.memory) > agent.batch_size * 2:
>         loss = agent.train_model()
>         losses.append(loss)
>         agent.update_target()
>메모리가 충분히 쌓이면 episode가 종료된 후 네트워크를 업데이트 합니다.

