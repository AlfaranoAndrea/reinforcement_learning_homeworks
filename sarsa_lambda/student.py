import numpy as np
import random


def epsilon_greedy_action(env, Q, state, epsilon):    
    num= random.random()
    if num < epsilon:
       action = env.action_space.sample()  # Explore action space       
     #  print("explore")
    else:
        action = np.argmax(Q[state])
    #    print(f"exploit, action={action}, vector= {Q[state]}")
 #   print(action)
    return action



def sarsa_lambda(env, alpha=0.2, gamma=0.99, lambda_= 0.8, initial_epsilon=1, n_episodes=100000 ):

    ####### Hyperparameters
    # alpha = learning rate
    # gamma = discount factor
    # lambda_ = elegibility trace decay
    # initial_epsilon = initial epsilon value
    # n_episodes = number of episodes

    ############# keep this shape for the Q!
    Q = np.zeros((env.observation_space.n, env.action_space.n)) 
    #Q = np.random.rand(env.observation_space.n, env.action_space.n)*0
    #np.random.rand(env.observation_space.n, env.action_space.n)

   
    # init epsilon
    epsilon = initial_epsilon
    received_first_reward = False

    print("TRAINING STARTED")
    print("...")
    for ep in range(n_episodes):
     
        eligibility=np.zeros((env.observation_space.n, env.action_space.n))
        state, _ = env.reset()
        action = epsilon_greedy_action(env, Q, state, epsilon)
        done = False
        while not done:
            ############## simulate the action
            next_state, reward, done, info, _ = env.step(action)
            next_action = epsilon_greedy_action(env, Q, next_state, epsilon)
            delta= reward + (gamma*Q[next_state][next_action])-Q[state][action]
            #eligibility[state]*= 0.9

            # eligibility[state][action] +=1
            
            # for i in range(0,4):
            #     if(i!=action):
            #         eligibility[state][i]*= 0.9 #[0,0, 0, 0]
            #eligibility[state]= [0,0, 0, 0]
            #eligibility[state][action] =1
            eligibility[state][action]+=1
            
            #eligibility[state]=[0,0, 0, 0]
            #eligibility[state][action]=1 
            Q= Q+  alpha*delta*eligibility
            eligibility*= lambda_*gamma


            if not received_first_reward and reward > 0:
                received_first_reward = True
                print("Received first reward at episode ", ep)

            # update current state and action
            state = next_state
            action = next_action

        # update current epsilon
        if received_first_reward:
            epsilon = 0.99 * epsilon
    
    print("TRAINING FINISHED")
    return Q