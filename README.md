# Double-DQN_theano_only

Author : Hugo Caselles-Dupré & Geoffrey Chinot

Disclaimer : several syntax errors can exist in the code. Use it wisely.

---------------------------------------
Project for ENSAE's ELTDM course : implementation of Deep Q-Network using only Theano. 
---------------------------------------

In this directory you can find a Python code with the implementation of Double DQN using only Theano, and another version using where Kears on top of Theano is used. The class Experience Replay allows you train the Double DQN agent on any OpenAI Gym environment.

An example, CartPole-v0, is provided. In double_dqn_theano.py, the neural network is build with Theano, and the code is compatible with the use of GPU via CUDA. In double_dqn_keras.py, the neural network is defined with Keras, adn the code is also compatible with the use of GPU via CUDA.

------------------------------------------

If you have any questions : casellesdupre.hugo@gmail.com.
