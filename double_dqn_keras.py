#EXPERIENCE REPLAY
import gym
import theano
import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import json
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras.models import model_from_json
import time
import operator


class ExperienceReplay(object):
    
    
    def __init__(self, max_memory=100, discount=.9):
        
        """Define max length of memory and gamma"""
        
        
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        
        
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        """Add experience to memory"""
        
        
        self.memory.append([states, game_over])
        #Delete the first experience if the memory is too long
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, target_model, model, batch_size=10):
        
        
        """Get the batch input and targets we will train on
        
        Uses Double DQN : http://www.aaai.org/Conferences/AAAI/2016/Papers/12vanHasselt12389.pdf """
        
        
        
        #length of memory vector
        len_memory = len(self.memory)
        
        #number of actions in action space
        num_actions = model.output_shape[-1]
        
        #states is an experience : [input_t_minus_1, action, reward, input_t],
        #so memory[0] is state and memory[0][0][0].shape[1] is the size of the input
        env_dim = self.memory[0][0][0].shape[1]
        
        #if batch_size<len_memory (it is mostly the case), 
        #then input is a matrix with batch_size rows and size of obs columns
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        
        #targets is a matrix with batch_size rows and number of actions columns
        targets = np.zeros((inputs.shape[0], num_actions))
        
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            
            #get experience number idx, idx being a random number in [0,length of memory]
            #There are batch_size experiences that are drawn
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            
            #Is the game over ? AKA done in gym
            game_over = self.memory[idx][1]

            #The inputs of the NN are the state of the experience drawn
            inputs[i:i+1] = state_t
            
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            # model.predict(state_t)[0] is the vector of Q(state_t) for each action
            targets[i] = target_model.predict(state_t)[0]
            
            #Q_sa=Q_target(s,argmax_a'{Q(s',a')}
            #index is the action you that maximizes the Q-value of the current network
            index, maxima = max(enumerate(model.predict(state_tp1)[0]), key=operator.itemgetter(1))
            #We take the value of the target network for action index
            Q_sa = target_model.predict(state_tp1)[0][index]
            
            
            # if game_over is True then the sequence is terminated 
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # the target for this particular experience is : reward_t + gamma * max_a' Q(s', a')
                # We know that you should have : 
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

# # CartPole on OpenAI Gym

# In[5]:

#Define the environment
env = gym.make('CartPole-v0')

#PARAMETERS

#learning rate
learning_rate=0.001

#exploration parameter
epsilon = .1

#decay rate for epsilon
decay_rate=0.85

#Number of possible actions
num_actions = env.action_space.n 

#Length of memory
max_memory = 400000

#Number of hidden units
hidden_size = 200

#Size of batch for training
batch_size = 50

#Accumulated reward over epoch
acc_reward=0

#shape of observations
observation_shape = env.observation_space.shape[0]

#counter of time-steps
time_step=0

#total number of time-steps to train on
max_time_steps=5000

#Number of times we update the target network
everyC=5

#start recording training part
#env.monitor.start('test',force=True,video_callable=lambda count: count % 50 == 0)
#env.monitor.configure(video_callable=lambda count: False)
#env.monitor.start('test',force=True)
                  
                  
#Parameter C
C=0

#RMSProp optimizer
#RMSprop=keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=1)
#Adam optimizer
Adam=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=1)

#Define the current DNN
model = Sequential()
#first fully connected layer, activation RELU
model.add(Dense(200, input_dim=observation_shape, activation='relu'))
#second fully connected layer, activation RELU
#model.add(Dense(200, activation='relu'))
#third fully connected layer, activation RELU
#model.add(Dense(200, activation='relu'))
#fourth fully connected layer, activation RELU
#model.add(Dense(60, activation='relu'))
#fourth fully connected layer, activation RELU
#model.add(Dense(12, activation='relu'))
#last fully connected layer, output Q(s,a,theta)
model.add(Dense(num_actions))
#choose optimization parameters
model.compile(optimizer=Adam, loss='mean_squared_error')

#Define the target DNN
target_model = Sequential()
#first fully connected layer, activation RELU
target_model.add(Dense(200, input_dim=observation_shape, activation='relu'))
#second fully connected layer, activation RELU
#target_model.add(Dense(200, activation='relu'))
#third fully connected layer, activation RELU
#target_model.add(Dense(200, activation='relu'))
#fourth fully connected layer, activation RELU
#target_model.add(Dense(60, activation='relu'))
#fourth fully connected layer, activation RELU
#target_model.add(Dense(12, activation='relu'))
#last fully connected layer, output Q(s,a,theta)
target_model.add(Dense(num_actions))
#choose optimization parameters
target_model.compile(optimizer=Adam, loss='mean_squared_error')

# If you want to continue training from a previous model, just uncomment the line bellow
#model.load_weights("model_cartpole")
#target_model.load_weights("model_cartpole")


# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

#nb of games won
win_cnt = 0

#start of traning time
t0 = time.time()

#actual training time
actual_total=0

#nb of episodes
e=0

print('Parameters :','epsilon :', epsilon,'C :', everyC,', learning rate :', learning_rate, 'batch size for training :', batch_size)

print(model.summary())

print('Training for ',max_time_steps,'time-steps ...')


#for e in range(epoch):
while time_step<max_time_steps:
    #set loss to zero
    loss = 0.
    
    #set accumulated reward to 0
    acc_reward = 0
    
    #Set C to zero
    C=0

    #add episode
    e+=1
    
    #env.reset() : reset the environment, get first observation
    input_t = env.reset()
    input_t = input_t.reshape((1,observation_shape))
    
    #the game starts, so set game_over to False
    game_over = False
    
    #Decay of epsilon
    #if e%100==0:
        #epsilon = epsilon*decay_rate
        #if epsilon<0.05:
            #epsilon=0.05
        #print('decay on epsilon')


    #decay on learning rate
    #if e%100==0:
        #learning_rate=learning_rate*(decay_rate**3)
        #if learning_rate<0.0001:
            #learning_rate=0.0001
        #Adam=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=1)
        #model.compile(optimizer=Adam, loss='mean_squared_error')
        #print('decay on learning rate')

    while not game_over:
        
        #set this state to be the last state
        input_tm1 = input_t
        
        # get next action according to espilon-greedy policy
        if np.random.rand() <= epsilon:
            #exploration
            action = np.random.randint(0, num_actions, size=1)[0]
        else:
            #exploitation
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

        #apply action, get rewards and new state
        input_t, reward, game_over, infodemerde = env.step(action)
        input_t = input_t.reshape((1,observation_shape))
        
        
        #Accumulate reward
        acc_reward += reward

        # store experience
        exp_replay.remember([input_tm1, action, reward, input_t], game_over)
        
        #Create new target network every C updates, by cloning the current network
        if C%everyC==0:
            model.save_weights("model_cartpole_TARGET", overwrite=True)
            with open("model_cartpole_TARGET.json", "w") as outfile:
                json.dump(model.to_json(), outfile) 
            target_model.load_weights("model_cartpole_TARGET")
            #print('LAAAA')
            
        #Increment C
        C += 1
        
        # get batch we will train on
        inputs, targets = exp_replay.get_batch(target_model, model, batch_size=batch_size)


        #start of actual training time
        t2 = time.time()

        #TRAIN
        loss += model.train_on_batch(inputs, targets)

        #end of actual training time
        t3 = time.time()
        actual_total += t3-t2

        #increment time-step
        time_step += 1

        #end game if max score is reached
        if acc_reward>=200:
            game_over=True
            win_cnt+=1


    #print("Epoch {:03d}/999 | Loss {:.4f} | Accumulated reward {:.4f}".format(e, loss, acc_reward))
    print(time_step,'time steps done, ',e,'episodes done. Reward :', acc_reward, ', loss :', loss)

#end of training time
t1 = time.time()
total = t1-t0
print('Total training time :', total,'Actual training time :', actual_total)
print('Win ratio (nb of games won/nb of games played) :', win_cnt/e)


#env.monitor.close()

#nb of episodes to test
nb_e_test=10

#Total reward over the episodes
total_rew=0


print('Testing for ',nb_e_test,'episodes ...')


for episode in range(nb_e_test):

    
    #set accumulated reward to 0
    acc_reward = 0
    
    
    #env.reset() : reset the environment, get first observation
    input_t = env.reset()
    input_t = input_t.reshape((1,observation_shape))
    
    #the game starts, so set game_over to False
    game_over = False
    

    while not game_over:
        
        #set this state to be the last state
        input_tm1 = input_t
        
        # get next action according to espilon-greedy policy
        # only exploitation
        q = model.predict(input_tm1)
        action = np.argmax(q[0])

        #apply action, get rewards and new state
        input_t, reward, game_over, infodemerde = env.step(action)
        input_t = input_t.reshape((1,observation_shape))
        
        
        #Accumulate reward
        acc_reward += reward


        #end game if max score is reached
        if acc_reward>=200:
            game_over=True

    #Total reward over the episodes
    total_rew+=acc_reward



    #print("Epoch {:03d}/999 | Loss {:.4f} | Accumulated reward {:.4f}".format(e, loss, acc_reward))
    print(episode,'episodes done. Reward :', acc_reward)


print('The average reward over the test was :',total_rew/nb_e_test)





