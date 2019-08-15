# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q_values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q_values[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        list_of_actions = self.getLegalActions(state)

        if not list_of_actions:
            return 0

        q_best = sys.maxint * -1

        for action in list_of_actions:
            temp = max(q_best, self.getQValue(state, action))
            q_best = temp
        return q_best

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        list_of_actions = self.getLegalActions(state)

        q_best = sys.maxint * -1
        best_action = None

        for action in list_of_actions:
            temp = self.getQValue(state, action)
            if temp > q_best:
                q_best = temp
                best_action = action
        return best_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action

        "*** YOUR CODE HERE ***"
        list_of_actions = self.getLegalActions(state)
        prob = self.epsilon

        if not list_of_actions:
            return None

        if util.flipCoin(prob):
            return random.choice(list_of_actions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        list_next_actions = self.getLegalActions(nextState)
        alpha = self.alpha
        gamma = self.discount
        qval = self.getQValue(state, action)
        best_next_qval = sys.maxint * -1

        for next_action in list_next_actions:
            temp = max(best_next_qval, self.getQValue(nextState, next_action))
            best_next_qval = temp

        if not list_next_actions:
            self.Q_values[(state, action)] = (1 - alpha) * qval + alpha * reward
        else:
            self.Q_values[(state, action)] = (1 - alpha) * qval + alpha * (reward + (gamma * best_next_qval))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q_val = 0
        list_of_features = self.featExtractor.getFeatures(state, action)
        for feature in list_of_features:
            # Q(state,action) = w * featureVector
            w = self.weights[feature]
            featureVector = list_of_features[feature]
            q_val += w * featureVector

        return q_val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        list_of_features = self.featExtractor.getFeatures(state, action)
        list_next_actions = self.getLegalActions(nextState)
        alpha = self.alpha
        gamma = self.discount
        qval = self.getQValue(state, action)
        best_next_qval = sys.maxint * -1

        for next_action in list_next_actions:
            temp = max(best_next_qval, self.getQValue(nextState, next_action))
            best_next_qval = temp

        for feature in list_of_features:
            if not list_next_actions:
                temp = self.weights[feature] + alpha * (reward - qval) * list_of_features[feature]
                self.weights[feature] = temp
            else:
                temp = self.weights[feature] + alpha * ((reward + gamma * best_next_qval) - qval) * \
                                        list_of_features[feature]
                self.weights[feature] = temp

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
