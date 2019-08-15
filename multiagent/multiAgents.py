# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentFood = currentGameState.getFood()

        "*** YOUR CODE HERE ***"

        """
        To so this you need to find out 3 separate things
        1. Where the closest food is and the distance to it
        2. Does the newPos share a location with a ghost and if so is that ghost not scared
        3. If stop is the action
        
        ^
        |
        |
        -The above code will be instrumental and finding the write solution
        Note: I have added Current food to it because it seems like that is what you want and not the food
        from the new successors...  
        """

        import sys  # sys has the max int like infinity in python
        maxDistance = sys.maxint * -1  # initialize the distance as the worst possible solution.

        """
        Part One:
        """
        for food in currentFood.asList():  # had to make a list so I could loop through.
            distance = (manhattanDistance(food, newPos)) * -1
            temp = max(distance, maxDistance)
            maxDistance = temp

        """
        Part Two:
        """
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos and (ghost.scaredTimer == 0):
                maxDistance = sys.maxint * -1

        """
        Part Three:
        """
        if action is 'Stop':
            maxDistance = sys.maxint * -1

        return maxDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        import sys

        """
        The simplest way to run minimax is to use recursion with 3 functions (minimax function, maxlevel function and 
        minlevel function).
        1. minimax function is used to check a states agent as well as if it is terminal or not. 
        2. max is used if the agent is pacman
        3. min is used if it is a ghost
        """

        """
        Part One
            a. need to check to see what agent we are on and if needed start the loop over. need to move the counter 
               along so that each agent is seen and then once it is it then starts back at pacman
            b. Need to check to see if it is terminal (Game is won or lost or if the depth is reached. 
            c. look at the agent and then from there go into part B or C depending if it is a pacman or ghost. 
        """

        def minimax(gameState, depth, agent):

            totAgents = gameState.getNumAgents()
            """
            Part A:
            """
            if agent >= totAgents:
                agent = 0  # back to pacman agent
                depth += 1  # it also means that the depth has increased by one

            """
            Part B:
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            """
            Part C:
            """
            if agent == 0:  # Pacman / maxLevel
                return maxLevel(gameState, depth, agent)
            else:  # ghost / minLevel
                return minLevel(gameState, depth, agent)

        """
        Part Two:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def maxLevel(gameState, depth, agent):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)
            """
            Part B.
            """
            value = [None, sys.maxint * -1]  # temp value for returning! should be an action and a value
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = minimax(childState, depth, agent + 1)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]
                if outValue > value[1]:
                    value = [action, outValue]
            return value

        """
        Part Three:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def minLevel(gameState, depth, agent):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)
            """
            Part B.
            """
            value = [None, sys.maxint]  # temp value for returning! should be an action and a value
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = minimax(childState, depth, agent + 1)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]
                if outValue < value[1]:
                    value = [action, outValue]
            return value

        """ This is where the function goes when orginally called. Then it will go up and into the helper functions
        that are nested into the function """
        startState = gameState
        startAgent = 0
        startDepth = 0

        value = minimax(startState, startAgent, startDepth)
        return value[0]  # need to return the game state from the minimax output. This is it!


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        import sys

        """
        Part One
            a. need to check to see what agent we are on and if needed start the loop over. need to move the counter 
               along so that each agent is seen and then once it is it then starts back at pacman
            b. Need to check to see if it is terminal (Game is won or lost or if the depth is reached. 
            c. look at the agent and then from there go into part B or C depending if it is a pacman or ghost. 
        """

        def minimax(gameState, depth, agent, alpha, beta):

            totAgents = gameState.getNumAgents()
            """
            Part A:
            """
            if agent >= totAgents:
                agent = 0  # back to pacman agent
                depth += 1  # it also means that the depth has increased by one

            """
            Part B:
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            """
            Part C:
            """
            if agent == 0:  # Pacman / maxLevel
                return maxLevel(gameState, depth, agent, alpha, beta)
            else:  # ghost / minLevel
                return minLevel(gameState, depth, agent, alpha, beta)

        """
        Part Two:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def maxLevel(gameState, depth, agent, alpha, beta):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)
            """
            Part B.
            """
            value = [None, sys.maxint * -1]  # temp value for returning! should be an action and a value
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = minimax(childState, depth, agent + 1, alpha, beta)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]
                if outValue > value[1]:
                    value = [action, outValue]

                """ NEW SECTION"""
                if outValue > beta:
                    value = [action, outValue]
                    return value
                temp = max(outValue, alpha)
                alpha = temp
                """ End of New"""

            return value

        """
        Part Three:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def minLevel(gameState, depth, agent, alpha, beta):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)
            """
            Part B.
            """
            value = [None, sys.maxint]  # temp value for returning! should be an action and a value
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = minimax(childState, depth, agent + 1, alpha, beta)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]
                if outValue < value[1]:
                    value = [action, outValue]

                """ NEW SECTION"""
                if outValue < alpha:  # time to trim!
                    value = [action, outValue]
                    return value
                temp = min(outValue, beta)  # time to update!
                beta = temp
                """ End of New"""

            return value

        """ This is where the function goes when orginally called. Then it will go up and into the helper functions
        that are nested into the function """
        startState = gameState
        startAgent = 0
        startDepth = 0
        startAlpha = sys.maxint * -1
        startBeta = sys.maxint

        finalTemp = minimax(startState, startAgent, startDepth, startAlpha, startBeta)
        return finalTemp[0]  # need to return the game state from the minimax output. This is it!


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        import sys

        """
        Part One
            a. need to check to see what agent we are on and if needed start the loop over. need to move the counter 
               along so that each agent is seen and then once it is it then starts back at pacman
            b. Need to check to see if it is terminal (Game is won or lost or if the depth is reached. 
            c. look at the agent and then from there go into part B or C depending if it is a pacman or ghost. 
        """

        def expminimax(gameState, depth, agent):

            totAgents = gameState.getNumAgents()
            """
            Part A:
            """
            if agent >= totAgents:
                agent = 0  # back to pacman agent
                depth += 1  # it also means that the depth has increased by one

            """
            Part B:
            """
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            """
            Part C:
            """
            if agent == 0:  # Pacman / maxLevel
                return maxLevel(gameState, depth, agent)
            else:  # ghost / minLevel
                return minLevel(gameState, depth, agent)

        """
        Part Two:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def maxLevel(gameState, depth, agent):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)
            """
            Part B.
            """
            value = [None, sys.maxint * -1]  # temp value for returning! should be an action and a value
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = expminimax(childState, depth, agent + 1)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]
                if outValue > value[1]:
                    value = [action, outValue]
            return value

        """
        Part Three:
            a. Check to see if terminal
            b. open up actions
            c. recursion on acctions
            d. take value given at actions and return min or max depenending on which level
        """

        def minLevel(gameState, depth, agent):

            agentActionList = gameState.getLegalActions(agent)  # list of the actions that the agent can take

            """
            Part A. 
            """
            if len(agentActionList) == 0:
                return self.evaluationFunction(gameState)

            """
            Part B.
            """
            value = [None, sys.maxint]  # temp value for returning! should be an action and a value
            temp = 0
            for action in agentActionList:
                childState = gameState.generateSuccessor(agent,
                                                         action)  # this built in function generates the possible
                # children after an action (foresight)
                """
            Part C.
                """
                childValues = expminimax(childState, depth, agent + 1)

                """
            Part D.
                """
                if type(childValues) is float:
                    outValue = childValues
                else:
                    outValue = childValues[1]

                """
                NEW SECTION
                    probability of action is just action/# of actions
                    Need to add them all up and associate them with an action
                    temp initially is 0 and then the probability of each action is added
                """
                temp += outValue / len(agentActionList)
                value = [action, temp]
                """------------------------------------------------------------------------------"""
            return value

        """ This is where the function goes when orginally called. Then it will go up and into the helper functions
        that are nested into the function """
        startState = gameState
        startAgent = 0
        startDepth = 0

        finalTemp = expminimax(startState, startAgent, startDepth)
        return finalTemp[0]  # need to return the game state from the minimax output. This is it!


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Tired to mimic what I did in question One and use ghost states and closest distance to food.
      I don't need the stop action any more but I still used the ghost distance and food distance to add them to the
      current score
    """
    "*** YOUR CODE HERE ***"

    ghostStates = currentGameState.getGhostStates()
    newPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()

    import sys  # sys has the max int like infinity in python
    maxDistance = sys.maxint * -1  # initialize the distance as the worst possible solution.

    dangerDistance = sys.maxint
    for ghost in ghostStates:
        danger = (manhattanDistance(ghost.getPosition(), newPos))
        temp = min(dangerDistance, danger)
        dangerDistance = temp

    """
    This is in case their is no food
    """
    if len(currentFood.asList()) == 0:
        return currentGameState.getScore() + dangerDistance
    """
    Loops through the food and finds the closest food. 
    """
    for food in currentFood.asList():  # had to make a list so I could loop through.
        distance = (manhattanDistance(food, newPos)) * -1
        temp = max(distance, maxDistance)
        maxDistance = temp

    return maxDistance + currentGameState.getScore() + dangerDistance


# Abbreviation
better = betterEvaluationFunction
