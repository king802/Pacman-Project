# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        import sys

        iterations = range(self.iterations)
        # loop thorugh the time steps
        for step in iterations:
            # need to copy list of value because we will over write the other list.
            list_of_values = self.values.copy()
            # loop through all the states at each time step
            for state in self.mdp.getStates():
                # Terminal Check
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    # VALUE ITERATION TIME!
                    # V_k(s) = MAX_a of the summation of all the s' over T(s,a,s')[R(s,a,s') + (Gamma)(Value(s')]
                    best_value = sys.maxint * -1
                    list_of_actions = self.mdp.getPossibleActions(state)
                    for action in list_of_actions:
                        list_of_transitions = self.mdp.getTransitionStatesAndProbs(state,
                                                                                   action)  # = list of (nextState, prob)
                        value = 0
                        for transition in list_of_transitions:
                            (nextState, prob) = transition
                            reward = self.mdp.getReward(state, action, nextState)
                            gamma = self.discount
                            value += prob * (reward + gamma * list_of_values[nextState])
                        best_value = max(value, best_value)

                    self.values[state] = best_value # this line will be rewritten for each itteration.

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s,a) = summation of all the s' over T(s,a,s')[R(s,a,s') + (Gamma)(Value(s')]

        # setting up the summation
        Q_value = 0

        list_of_transitions = self.mdp.getTransitionStatesAndProbs(state, action)  # = list of (nextState, prob)
        for transition in list_of_transitions:
            (nextState, prob) = transition  # nextState = s' | prob is T(s,a,s')
            reward = self.mdp.getReward(state, action, nextState)  # R(s,a,s')
            gamma = self.discount  # Gamma = gamma | this is the discount factor

            # Q(s,a) = summation of all the s' over T(s,a,s')[R(s,a,s') + (Gamma)(Value(s')]
            Q_value += prob * (reward + (gamma * self.values[nextState]))
        return Q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # this is policy iteration and we can use the Q(s,a) to compute it.

        """!!You may break ties any way you see fit.  Note that if
                  there are no legal actions, which is the case at the
                  terminal state, you should return None!!"""
        if self.mdp.isTerminal(state):
            return None
        import sys

        best_value = sys.maxint * -1  # cannot just have it be zero because values could be negative.
        list_of_actions = self.mdp.getPossibleActions(state)
        for action in list_of_actions:
            value = self.computeQValueFromValues(state, action)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
