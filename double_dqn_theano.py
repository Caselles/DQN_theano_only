import sys
import timeit
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

''' Definition of the neural network with Theano '''


def relu(X):
    X[numpy.where(X < 0)] = 0
    return(X)


class output_layer(object):

    def __init__(self, input, n_in, n_out):
        
 
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.y_pred = T.dot(input, self.W) + self.b

        self.params = [self.W, self.b]

        self.input = input

    def mse(self, y):
        return T.mean((self.y_pred - y) ** 2) 
    
    
class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        self.input = input
        if W is None:
            W_values = numpy.asarray(
                numpy.random.uniform(low=-0.1,high=0.1,size=(n_in,  n_out)),
                dtype=theano.config.floatX)
            W_h = theano.shared(value = W_values, name='W_h', borrow=True)
            
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b_h = theano.shared(value=b_values, name='b_h', borrow=True)

        self.W_h = W_h
        self.b_h = b_h
        
        self.params = [self.W_h, self.b_h] # Parameters of the model
        
        self.output = T.nnet.relu(T.dot(input, self.W_h) + self.b_h)

        self.input = input
    
    
class Neural_Network(object):
    
    def __init__(self, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            input=input,
            n_in= n_in,
            n_out=n_hidden,
        )

        self.output_layer = output_layer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )


        self.mse = (
            self.output_layer.mse
        )

        self.params = self.hiddenLayer.params + self.output_layer.params
        
        self.input = input
        
def train_on_batch(dataset_X, dataset_y, classifier, x,y,index,learning_rate=0.01, batch_size=1, n_hidden=500):

    train_set_x = theano.shared(numpy.asarray(dataset_X,
                                            dtype=theano.config.floatX),
                                borrow=True)
    train_set_y = theano.shared(numpy.asarray(dataset_y,
                                            dtype=theano.config.floatX),
                                borrow=True)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size



    cost = (
        classifier.mse(y)
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]


    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    #print('... training')
    
    start_time = timeit.default_timer()

    error = []
    error_min = numpy.inf
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        linear_1 = numpy.dot(dataset_X, classifier.hiddenLayer.W_h.eval()) +  classifier.hiddenLayer.b_h.eval()
        h1 = relu(linear_1)
        output_nn = numpy.dot(h1,classifier.output_layer.W.eval()) + classifier.output_layer.b.eval()
        error.append(numpy.mean((output_nn - dataset_y)**2))
        
        '''if error[len(error)-1] < error_min:
            error_min = error[len(error)-1]
            best_W1 = classifier.hiddenLayer.W_h.eval()
            best_b1 = classifier.hiddenLayer.b_h.eval()
            best_Wo = classifier.output_layer.W.eval()
            best_bo = classifier.output_layer.b.eval()'''

    end_time = timeit.default_timer()
    
    return(error, classifier.hiddenLayer.W_h.eval(), classifier.hiddenLayer.b_h.eval(), classifier.output_layer.W.eval(), classifier.output_layer.b.eval())

def predict_NN(X_input, W1, b1, Wo, bo):
    return(numpy.dot(relu(numpy.dot(X_input,W1) + b1),Wo) + bo)

''' Experience Replay '''


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

    def get_batch(self, W1, b1, Wo, bo, batch_size=10):
        
        
        """Get the batch input and targets we will train on
        
        Uses Double DQN : http://www.aaai.org/Conferences/AAAI/2016/Papers/12vanHasselt12389.pdf """
        
        
        
        #length of memory vector
        len_memory = len(self.memory)
        
        #number of actions in action space
        num_actions = 2
        
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
            targets[i] = predict_NN(state_t, W1, b1, Wo, bo)

            #Q_sa=max_a{Q}
            Q_sa = np.max(targets[i])
            
            
            # if game_over is True then the sequence is terminated 
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # the target for this particular experience is : reward_t + gamma * max_a' Q(s', a')
                # We know that you should have : 
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

''' CartPole on OpenAI Gym '''

#Define the environment
env = gym.make('CartPole-v0')

#PARAMETERS

#learning rate
learning_rate=0.1

#exploration parameter
epsilon = .1

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
everyC=50

#start recording training part
#env.monitor.start('test',force=True,video_callable=lambda count: count % 50 == 0)
#env.monitor.configure(video_callable=lambda count: False)
#env.monitor.start('test',force=True)
                  
                  
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


print('Training for ',max_time_steps,'time-steps ...')

N = 6
X = numpy.random.uniform(low=-5.,high=5.,size=(N, 4)).astype('float32')
W = numpy.random.uniform(low=-5.,high=5.,size=(4, 2)).astype('float32')
b = numpy.random.uniform(low=-5.,high=5.,size=2).astype('float32') 

noise = numpy.random.normal(0,1,(N,2))

y = numpy.dot(X**2,W) + 5*numpy.dot(X,W) +  b + noise
y=y.astype('float32')


# allocate symbolic variables for the data
index = T.lscalar()  
x = T.matrix('x') 
y_g = T.matrix('y')  
# construct the neural net
classifier = Neural_Network(input=x,n_in=4,n_hidden=200,n_out=2)

#initalize with a random backpropagation
err, W1, b1, Wo, bo = train_on_batch(X, y, classifier, x=x,y=y_g,index=index, learning_rate=0.0003, batch_size=N)

#Training the algorithm

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
        input_tm1 = input_t.astype('float32')
        
        # get next action according to espilon-greedy policy
        if np.random.rand() <= epsilon:
            #exploration
            action = np.random.randint(0, num_actions, size=1)[0]
        else:
            #exploitation
            q = predict_NN(input_tm1, W1, b1, Wo, bo)
            action = np.argmax(q)

        #apply action, get rewards and new state
        input_t, reward, game_over, infodemerde = env.step(action)
        input_t = input_t.reshape((1,observation_shape))
        
        
        #Accumulate reward
        acc_reward += reward

        # store experience
        exp_replay.remember([input_tm1, action, reward, input_t], game_over)
        
        #Create new target network every C updates, by cloning the current network
        '''if C%everyC==0:
            model.save_weights("model_cartpole_TARGET", overwrite=True)
            with open("model_cartpole_TARGET.json", "w") as outfile:
                json.dump(model.to_json(), outfile) 
            target_model.load_weights("model_cartpole_TARGET")
            
        #Increment C
        C += 1'''
        
        # get batch we will train on
        inputs, targets = exp_replay.get_batch(W1, b1, Wo, bo, batch_size=batch_size)
        inputs=inputs.astype('float32')
        targets=targets.astype('float32')


        #start of actual training time
        t2 = time.time()

        #TRAIN
        err, W1, b1, Wo, bo = train_on_batch(inputs, targets, classifier, x=x, y=y_g, index=index, learning_rate=learning_rate, batch_size=len(inputs))

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
    print(time_step,'time steps done, ',e,'episodes done. Reward :', acc_reward, ', loss :', err[0])

#end of training time
t1 = time.time()
total = t1-t0
print('Total training time :', total,'Actual training time :', actual_total)
print('Win ratio (nb of games won/nb of games played) :', win_cnt/e)

#Testing the algorithm

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
        input_tm1 = input_t.astype('float32')
        
        # get next action according to espilon-greedy policy
        # only exploitation
        q = predict_NN(input_tm1, W1, b1, Wo, bo)
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

