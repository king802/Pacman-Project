# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    fringe = util.Stack()
    visited_states = []

    initial_state = (problem.getStartState(), 0, [])

    fringe.push(initial_state)
    while not fringe.isEmpty():
        state = fringe.pop()
        (location, cost, path) = state

        if location not in visited_states:
            visited_states.append(location)
            if problem.isGoalState(location):
                return path
            children = problem.getSuccessors(location)
            for child in children:
                if child[0] not in visited_states:
                    fringe.push((child[0], 0, path + [child[1]]))
    return []

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = util.Queue()  # BFS uses a FIFO method so a stack will be used
    visited_states = []  # Need this for tree search!
    # Initial State (successor, action (empty), stepCost (0 @start))
    initial_state = (problem.getStartState(), [], 0)

    # Add the first node to the fringe!
    fringe.push(initial_state)

    # pop into a current state
    current_state = fringe.pop()
    (state, lod, cost) = current_state

    # need to keep track of this so its a tree and not a graph search!
    visited_states.append(state)

    # time to expand and fill and pop from the fringe!
    while not problem.isGoalState(state):
        succ = problem.getSuccessors(state)
        for x in succ:
            if not x[0] in visited_states:  # Tree Search Logic! No Loops!
                new_state = (x[0], lod + [x[1]], cost + x[2])  # Gets new state from successor
                fringe.push(new_state)  # Adds new successor to the fringe to then be removed in BFS order
                visited_states.append(x[0])  # More Tree Search functionality
        current_state = fringe.pop()  # Pops the next on the thing for BFS to search
        (state, lod, cost) = current_state  # gives the state that is to be searched values to test against
    return lod
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # need to keep track of cost
    visited_states = {}  # Need this for tree search!
    # Initial State (successor, action (empty), stepCost (0 @start))
    initial_state = (problem.getStartState(), [], 0)

    # Add the first node to the fringe!
    fringe.push(initial_state, 0)

    # pop into a current state
    current_state = fringe.pop()
    (state, lod, cost) = current_state

    # need to keep track of this so its a tree and not a graph search!
    visited_states[state] = cost

    # time to expand and fill and pop from the fringe!
    while not problem.isGoalState(state):
        succ = problem.getSuccessors(state)
        for x in succ:
            if not (visited_states.has_key(x[0]) and visited_states[x[0]] < (x[2] + cost)):
                new_state = (x[0], lod + [x[1]], cost + x[2])  # Gets new state from successor
                fringe.push(new_state, new_state[2])  # Adds new successor to the fringe to then be removed in BFS order
                visited_states[new_state[0]] = new_state[2]  # More Tree Search functionality
        current_state = fringe.pop()  # Pops the next on the thing for BFS to search
        (state, lod, cost) = current_state  # gives the state that is to be searched values to test against
    return lod

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()  # need to keep track of cost
    visited_states = []
    # Initial State (successor, action (empty), stepCost (0 @start))
    initial_state = (problem.getStartState(), [], 0)

    # Add the first node to the fringe!
    fringe.push(initial_state, 0 + heuristic(initial_state[0], problem))

    # pop into a current state
    current_state = fringe.pop()
    (location, path, cost) = current_state

    # need to keep track of this so its a tree and not a graph search!
    delta = cost + heuristic(location, problem)

    visited_states.append((location, delta))

    # time to expand and fill and pop from the fringe!
    # while not problem.isGoalState(location):
    #     children = problem.getSuccessors(location)
    #     for child in children:
    #         child_cost = cost + child[2]
    #         for (visited_location, cost_at_locatio) in visited_states:
    #             if child[0]
    #         if not (visited_states.has_key(x[0]) and visited_states[x[0]] < (x[2] + cost + heuristic(x[0], problem))):
    #             new_state = (
    #                 x[0], lod + [x[1]], (cost + x[2]))
    #             fringe.push(new_state, new_state[2] + heuristic(new_state[0],
    #                                                             problem))
    #             visited_states[new_state[0]] = new_state[2] + heuristic(new_state[0],
    #                                                                     problem)
    #     current_state = fringe.pop()
    #     (state, lod, cost) = current_state
    # return lod


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
