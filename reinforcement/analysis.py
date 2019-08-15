# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise


def question3a():
    # Prefer the close exit (+1), risking the cliff (-10)
    answerDiscount = 0.1  # discount is big to make +10 small
    answerNoise = 0  # always follows poicy so it will take the scary rout
    answerLivingReward = -1  # wants to end the game
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    answerDiscount = .5  # half discount factor so it still prefers 1 over +10
    answerNoise = .4  # need some noise so it doesnt want to risk cliffs
    answerLivingReward = -1  # still do not want it to live forever
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3c():
    # Prefer the distant exit (+10), risking the cliff (-10)
    answerDiscount = 1  # no discount so +10 is good
    answerNoise = 0  # so it goes scary rout
    answerLivingReward = -1  # still need it to end
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    # Prefer the distant exit (+10), avoiding the cliff (-10)
    answerDiscount = 1  # do not want any discount on +10
    answerNoise = .5  # need noise to redirect it around cliff
    answerLivingReward = -.5  # not as step of a living penalty so it can make it to +10 and still get points
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    # Avoid both exits and the cliff (so an episode should never terminate)
    answerDiscount = 1
    answerNoise = 0
    answerLivingReward = +1 # it doesnt want to end
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question6():
    answerEpsilon = 1
    answerLearningRate = None
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis

    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
