
import numpy as np
import sys
import math


ACTION_COST = {
    'A1': 1,
    'A2': 1.5,
    'A3': 0.5,
    'A4': 0.5
}

ACTION_SET = ['A1', 'A2', 'A3', 'A4']

Nonestate = np.zeros(3)

class Action:

    def __init__(self, action):
        self.action = action
        self.output = Nonestate
        self.cost = ACTION_COST[action]

    def __str__(self):
        return self.action

    
    def ValidAction(self, input, outstate):
        # outside grid
        if outstate[0] < 1 or outstate[0] > 5 or outstate[1] < 1 or outstate[1] > 5:
            return False
        # blocks 
        if (outstate[:2] == [3, 2]).all():
            return False

        if (outstate[:2] == [2, 5]).all():
            return False
        if (outstate[:2] == [4, 5]).all():
            return False

        # barriers
        if input[0] == outstate[0] == 1 and input[1] < 3 and outstate[1] >= 3:
            return False
        if input[0] == outstate[0] == 1 and input[1] >= 3 and outstate[1] < 3:
            return False
        if input[0] == outstate[0] == 1 and input[1] < 4 and outstate[1] >= 4:
            return False
        if input[0] == outstate[0] == 1 and input[1] >= 4 and outstate[1] < 4:
            return False
        if input[0] == outstate[0] == 5 and input[1] < 3 and outstate[1] >= 3:
            return False
        if input[0] == outstate[0] == 5 and input[1] < 3 and outstate[1] >= 3:
            return False
        if input[0] == outstate[0] == 5 and input[1] < 4 and outstate[1] >= 4:
            return False
        if input[0] == outstate[0] == 5 and input[1] < 4 and outstate[1] >= 4:
            return False
        return True

    def Move1(self, input):
        
        output = np.array(input)

        if input[2] == 1:  # up
            output[1] += 1
        elif input[2] == 2:  # down
            output[1] -= 1
        elif input[2] == 3:  # left
            output[0] -= 1
        elif input[2] == 4:  # right
            output[0] += 1

        if self.ValidAction(input, output):
            self.output = output
        else:
            self.output = Nonestate

    def Move2(self, input):

        output = np.array(input)

        if input[2] == 1:  # up
            output[1] += 2
        elif input[2] == 2:  # down
            output[1] -= 2
        elif input[2] == 3:  # left
            output[0] -= 2
        elif input[2] == 4:  # right
            output[0] += 2

        if self.ValidAction(input, output):
            self.output = output
        else:
            self.output = Nonestate

    def TurnLeft(self, input):
        
        output = np.array(input)
        # forward directions
        if input[2] == 1:  # up
            output[2] = 3
        elif input[2] == 2:  # down
            output[2] = 4
        elif input[2] == 3:  # left
            output[2] = 2
        elif input[2] == 4:  # right
            output[2] = 1

        self.output = output

        if self.ValidAction(input, output):
            self.output = output
        else:
            return

    def TurnRight(self, input):
       
        output = np.array(input)
        # forward directions
        if input[2] == 1:  # up
            output[2] = 4
        elif input[2] == 2:  # down
            output[2] = 3
        elif input[2] == 3:  # left
            output[2] = 1
        elif input[2] == 4:  # right
            output[2] = 2

        self.output = output
        if self.ValidAction(input, output):
            self.output = output
        else:
            return

    def Act(self, state):

        if self.action == "A1":
            self.Move1(state)
        elif self.action == "A2":
            self.Move2(state)
        elif self.action == "A3":
            self.TurnLeft(state)
        elif self.action == "A4":
            self.TurnRight(state)

        return self.output

stateSize = 5 * 5 * 4
actSize = 4

def FindVIndex(state):
    index = state - 1
    return index[0]*20 + index[1]*4 + index[2]

def FindState(index):
    x = index // 20 
    y = (index - x * 20)//4 
    ori = (index - x * 20)%4 

    return np.array([x,y,ori])


def Value_Iter(gamma, noise):
    Vcur = np.zeros(stateSize)
    Vnew = np.zeros(stateSize)

    Best_Action_noise = [[0.0] * actSize for i in range( stateSize)]
   
    Best_Action = ["None"] * stateSize
    
    for iter in range(100):
        
        # update Vnew for each state
        for currentState_idx, vi in enumerate(Vcur):
       
            current_state = FindState(currentState_idx)+1
            Q = np.zeros(actSize)
            # if current  is in the red
            if (current_state[:2] == [3,4]).all():
                #Vnew[currentState_idx] = -1000*gamma

                continue
            # if current  is in the green
            if (current_state[:2] == [5,5]).all():
                #Vnew[currentState_idx] = 100*gamma
                continue
            
            # build Q_value list for each state
            for action_idx, action in enumerate(ACTION_SET):

                current_action = Action(action)
                next_state = current_action.Act(current_state)

                nextState_idx = FindVIndex(next_state)

                # if act is 2 steps forward and go through red block
                if action == "A2":
                    # cross horizontally
                    if (current_state[1] == next_state[1] == 4) and (current_state[0]+next_state[0]==6):
                        continue
                    # cross vertically
                    if (current_state[0] == next_state[0]== 3) and (current_state[1]+next_state[1]==8):
                        continue

                # next state is an impossible state
                if (next_state == Nonestate).all():
                    Q[action_idx] = np.nan
                elif (next_state[:2] == [3,4]).all():
                    Q[action_idx] = -1000 + (-current_action.cost)+gamma*Vcur[nextState_idx]
                # next state is target
                elif (next_state[:2] == [5,5]).all():
                    Q[action_idx] = 100 + (-current_action.cost)+gamma*Vcur[nextState_idx]
                # normal grid
                else:
                    Q[action_idx] = (-current_action.cost)+gamma*Vcur[nextState_idx]

            # no noise
            if not noise :
                # Update Vi according to noise
                Vnew[currentState_idx] = np.nanmax(Q)
                # Update policy
                bestAction_index = np.nanargmax(Q)
                Best_Action[currentState_idx] = ACTION_SET[bestAction_index]

                # print first 3 iteration
                if iter in range(3):
                    print("iter "+str(iter)+":")
                    for  i in range(stateSize):
                        print("state " + str(FindState(i)+1) + " Value = " +str(Vcur[i])+ " action: "+ Best_Action[i])
            
            # with noise
            if noise:
                bestAction_index = np.nanargmax(Q)

                
                num_possible_actions = actSize - np.count_nonzero(np.isnan(Q))
                noise_probability  = noise/float(num_possible_actions-1.0)
                
                probability_array = np.zeros(actSize)
                probability_array[~np.isnan(Q)] = noise_probability

                probability_array[bestAction_index] = 1.0-noise

                Q_m = np.ma.array(Q, mask=np.isnan(Q))

                # Update V
              
                Vnew[currentState_idx] = np.ma.dot(Q_m,probability_array)
            
                for i in range(actSize):
                    
                    Best_Action_noise[currentState_idx][i] = probability_array[i]
             
        
        Vcur = np.copy(Vnew)
        
        iter = iter + 1

    if noise:
         print("100 ITERATION")
         for i in range(stateSize):
            print("state " + str(FindState(i)+1) + " V value: " +str(Vcur[i]))
            for a in range(actSize):
                print(str(ACTION_SET[a]) + " probability: " + str(Best_Action_noise[i][a]))

    if not noise:
        print("100 ITERATION")
        for i in range(stateSize):
            print("state " + str(FindState(i)+1) + " V value: " +str(Vcur[i])+ " action: "+ Best_Action[i])


if __name__ == '__main__':
    # gamma = float(sys.argv[1])
    # noise = float(sys.argv[2])

    gamma = 0.9
    noise = 0.1

    Value_Iter(gamma, noise)
