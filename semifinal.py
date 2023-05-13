from time import time
from vehicle import Driver
import torch.nn as nn
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline


from vehicle import Driver
from time import time

LIDAR_devices = [
    "front",
    "front right 0",
    "front right 1",
    "front right 2",
    "front left 0",
    "front left 1",
    "front left 2",
    "rear",
    "rear left",
    "rear right",
    "right",
    "left"]
all_Sensors = {}

road_lanes = [10.6, 6.875, 3.2]
currentLane = 1
overtakingSide = None
maxSpeed = 80
safeOvertake = False


def turn_Error(position, targetPosition):
    # print("position", position)
    """Apply the PID controller and return the angle command."""
    Power = 0.05
    Integral_ = 0.00015
    Differential = 25
    Error = position - targetPosition
    if turn_Error.previousError is None:
        turn_Error.previousError = Error
    # anti-windup mechanism
    if Error > 0 and turn_Error.previousError < 0:
        turn_Error.integral = 0
    if Error < 0 and turn_Error.previousError > 0:
        turn_Error.integral = 0
    turn_Error.integral += Error
    # compute angle
    angle = Power * Error + Integral_ * turn_Error.integral + \
        Differential * (Error - turn_Error.previousError)
    turn_Error.previousError = Error
    return angle


turn_Error.integral = 0
turn_Error.previousError = None


def get_filtered_speed(speed):
    """Filter the speed command to avoid abrupt speed changes."""
    get_filtered_speed.previousSpeeds.append(speed)
    if len(get_filtered_speed.previousSpeeds) > 100:  # keep only 80 values
        get_filtered_speed.previousSpeeds.pop(0)
    return sum(get_filtered_speed.previousSpeeds) / float(len(get_filtered_speed.previousSpeeds))


def is_vehicle_on_side(side):
    """Check (using the 3 appropriated front distance all_Sensors) if there is a car in front."""
    for i in range(3):
        name = "front " + side + " " + str(i)
        if all_Sensors[name].getValue() > 0.8 * all_Sensors[name].getMaxValue():
            return True
    return False


def reduce_speed_if_vehicle_on_side(speed, side):
    """Reduce the speed if there is some vehicle on the side given in argument."""
    minRatio = 1
    for i in range(3):
        name = "front " + overtakingSide + " " + str(i)
        ratio = all_Sensors[name].getValue() / all_Sensors[name].getMaxValue()
        if ratio < minRatio:
            minRatio = ratio
    return minRatio * speed


get_filtered_speed.previousSpeeds = []
driver = Driver()

for name in LIDAR_devices:
    all_Sensors[name] = driver.getDevice("distance sensor " + name)
    all_Sensors[name].enable(10)


gps = driver.getDevice("gps")
gps.enable(10)

camera = driver.getDevice("camera")
# uncomment those lines to enable the camera
# camera.enable(10)
# camera.recognitionEnable(50)

translation_super = driver.getSelf().getField('translation')
print(driver.getSelf().getField('translation'))
rotation_super = driver.getSelf().getField('rotation')
initial_tr = translation_super.getSFVec3f()
initial_rot = rotation_super.getSFRotation()

current_time = time()

# print(translation_super.getMFString, rotation_super, initial_tr, initial_rot)

# frontDistance = all_Sensors["front"].getValue()
# print(frontDistance)


def reset_car():
    l = random.choice([0, 1, 2])
    translation_super.setSFVec3f([initial_tr[0], road_lanes[l], initial_tr[2]])
    rotation_super.setSFRotation(initial_rot)
    driver.setCruisingSpeed(0.)
    driver.setBrakeIntensity(0)
    driver.getSelf().resetPhysics()
    state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             gps.getValues()[1]/(road_lanes[0] - road_lanes[2]), 1]
    return l


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(
            np.array(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            print(action_values)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # Run from Model
        if True:
            print('model')
            return np.argmax(action_values.cpu().data.numpy())
        # if random.random() > eps:
        #     print('model')
        #     return np.argmax(action_values.cpu().data.numpy())
        else:
            print('random')
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(
            next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


agent = Agent(state_size=14, action_size=7, seed=0)

actions = [0, 1, 2, 3, 4, 5, 6]
# def step(action):
# state = []
# for i in LIDAR_devices:
#     state.append((all_Sensors[str(i)].getValue())/20)
# state.append(gps.getValues()[1])

# done = False
# # Collision
# for k, v in all_Sensors.items():
#     if v.getValue() < 0.15 * v.getMaxValue():
#         reset_car()
#         done = True
#         wait_for_lane_change = False
# if done:
#     reward = -50
# else:
#     reward = 5
#     if previous_lane != lane:
#         reward = -1

# if -500 > gps.getValues()[0]:
#     reward = 50
#     reset_car()
#     done = True
#     wait_for_lane_change = False
# return state, reward, done


# watch an untrained agent
# state = reset_car()
# for j in range(200):
#     action = agent.act(state)
#     state, reward, done, _ = step(action)
#     if done:
#         break

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon


n_episodes = 2000
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995

n_actions = len(actions)
lane = 1
previous_lane = lane
wait_for_lane_change = False

t = 0
state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         gps.getValues()[1]/(road_lanes[0] - road_lanes[2]), 1]
score = 0

# scores = dqn()

matplotDict = {}
eps = eps_start

n_episode = 0
# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
scores = 0
while driver.step() != -1:
    scores = 0
    t += 1
    position = gps.getValues()[1]
    if lane != previous_lane and not wait_for_lane_change:
        wait_for_lane_change = True

    if wait_for_lane_change:
        if lane == 0:
            if 8 < position:
                wait_for_lane_change = False
                previous_lane = lane
        elif lane == 1:
            if 5 < position < 8:
                wait_for_lane_change = False
                previous_lane = lane
        elif lane == 2:
            if position < 5:
                wait_for_lane_change = False
                previous_lane = lane

    if not wait_for_lane_change:
        action = agent.act(state, eps)
        print(action)
        speed = round(driver.getCurrentSpeed())
        if action < 3:
            lane = action
        if speed > 5:
            if action == 3:
                speed = 5
            elif action == 4:
                speed = speed/(speed/10)
            elif action == 5:
                speed = speed/(speed/20)
            elif action == 6:
                speed = speed/(speed/30)
        else:
            speed = 40

    driver.setCruisingSpeed(speed)
    # print('  speed= ', driver.getCurrentSpeed())

    angle = max(
        min(turn_Error(gps.getValues()[1], road_lanes[lane]), 0.5), -0.5)
    driver.setSteeringAngle(-angle)

    state = []
    for i in LIDAR_devices:
        state.append((all_Sensors[str(i)].getValue())/20)
    state.append(gps.getValues()[1]/(road_lanes[0] - road_lanes[2]))
    state.append(round(driver.getCurrentSpeed()/30))

    done = False
    # Collision
    for k, v in all_Sensors.items():
        if v.getValue() < 0.15 * v.getMaxValue():
            lane = reset_car()
            score = 0
            eps = eps_start
            done = True
            wait_for_lane_change = False

    if done:
        reward = -50
    else:
        reward = 0.1
        if previous_lane != lane:
            reward -= 0.01

    for k, v in all_Sensors.items():
        if v.getValue() < 0.30 * v.getMaxValue():
            reward -= 0.01
    for k, v in all_Sensors.items():
        if v.getValue() < 0.50 * v.getMaxValue():
            reward -= 0.001
    for k, v in all_Sensors.items():
        if v.getValue() < 0.70 * v.getMaxValue():
            reward += 0.0001
    for k, v in all_Sensors.items():
        if v.getValue() < 0.90 * v.getMaxValue():
            reward += 0.001

    if -240 > gps.getValues()[0]:
        reward = 100
        lane = reset_car()
        done = True
        wait_for_lane_change = False

    # next_state, reward, done = step(action)
    next_state = []
    for i in LIDAR_devices:
        next_state.append((all_Sensors[str(i)].getValue())/20)
    next_state.append(gps.getValues()[1]/(road_lanes[0] - road_lanes[2]))
    next_state.append(round(driver.getCurrentSpeed()/30))

    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
        n_episode += 1
        with open('reward.csv', 'a') as f:
            text = ''
            text += '\n'
            text += f'{n_episode}, {reward}'
            f.write(text)
        distance = gps.getValues()[0]
        with open('distance.csv', 'a') as f1:
            text = ''
            text += '\n'
            text += f'{n_episode}, {distance}'
            f1.write(text)
        with open('speed.csv', 'a') as f2:
            text = ''
            text += '\n'
            text += f'{n_episode}, {speed}'
            f2.write(text)
        with open('action.csv', 'a') as f3:
            text = ''
            text += '\n'
            text += f'{n_episode}, {action}'
            f3.write(text)

        print(f'Episode {t}\tAverage Score: {np.mean(scores_window):.2f}')

        state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, gps.getValues(
        )[1]/(road_lanes[0] - road_lanes[2]), round(driver.getCurrentSpeed()/30)]
    scores_window = deque(maxlen=100)  # last 100 scores

    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps)  # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(
        t, np.mean(scores_window)), end="")
    if t % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            t, np.mean(scores_window)))

    if np.mean(scores_window) >= 200.0:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
            t-100, np.mean(scores_window)))
        # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        # break
    matplotDict[t] = reward


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
