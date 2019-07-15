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

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        dis2ghost = float("inf")
        for ghostState in newGhostStates:
            dis2ghost = min(dis2ghost, manhattanDistance(ghostState.getPosition(), newPos))
        if dis2ghost > 0:
            score -= 20 / dis2ghost
        if newFood.asList():
            dis2food = float("inf")
            for food in newFood.asList():
                dis2food = min(dis2food, manhattanDistance(food, newPos))
            score += 10 / dis2food
        return score


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

        def minimax(state, agentid, curdepth):
            # Build layer for each agent, if the agent is pacman, return max value, else return min value.
            if agentid != state.getNumAgents():
                acts = state.getLegalActions(agentid)
                if acts:
                    nextlayer = [minimax(state.generateSuccessor(agentid, act), agentid + 1, curdepth) for act in acts]
                    return max(nextlayer) if agentid == 0 else min(nextlayer)
                else:
                    return self.evaluationFunction(state)
            # Finished building layer for each agent
            # If haven't reached the setting depth, start from pacman.
            # If reached the desired depth, return the leaves value.
            else:
                if curdepth < self.depth:
                    return minimax(state, 0, curdepth + 1)
                else:
                    return self.evaluationFunction(state)

        # Return the next action which has the max score.
        next_act = ''
        maxscore = -10000
        for action in gameState.getLegalActions(0):
            tmpscore = minimax(gameState.generateSuccessor(0, action), 1, 1)
            if tmpscore > maxscore:
                maxscore = tmpscore
                next_act = action
        return next_act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agentid, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            score = float("-inf")
            for act in state.getLegalActions(agentid):
                next_score = min_value(state.generateSuccessor(agentid, act), agentid + 1, depth, alpha, beta)
                score = max(score, next_score)
                if beta and beta < score:
                    return score
                alpha = max(alpha, score)
            return score if score > float("-inf") else self.evaluationFunction(state)

        def min_value(state, agentid, depth, alpha, beta):
            if agentid%state.getNumAgents() == 0:
                return max_value(state, 0, depth + 1, alpha, beta)
            score = float("inf")
            for act in state.getLegalActions(agentid):
                next_score = min_value(state.generateSuccessor(agentid, act), agentid + 1, depth, alpha, beta)
                score = min(score, next_score)
                if alpha and alpha > score:
                    return score
                beta = min(beta, score) if beta else score
            return score if score < float("inf") else self.evaluationFunction(state)

        best_act = ''
        alpha1 = None
        beta1 = None
        for act in gameState.getLegalActions(0):
            score = min_value(gameState.generateSuccessor(0, act), 1, 1, alpha1, beta1)
            if not alpha1:
                alpha1 = score
                best_act = act
            else:
                if alpha1 < score:
                    alpha1 = score
                    best_act = act
        return best_act


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
        def expectimax(state, agentid, curdepth):
            # Build layer for each agent, if the agent is pacman, return max value, else return min value.
            if agentid != state.getNumAgents():
                acts = state.getLegalActions(agentid)
                if acts:
                    nextlayer = [expectimax(state.generateSuccessor(agentid, act), agentid + 1, curdepth) for act in acts]
                    return max(nextlayer) if agentid == 0 else sum(nextlayer)/len(nextlayer)
                else:
                    return self.evaluationFunction(state)
            # Finished building layer for each agent
            # If haven't reached the setting depth, start from pacman.
            # If reached the desired depth, return the leaves value.
            else:
                if curdepth < self.depth:
                    return expectimax(state, 0, curdepth + 1)
                else:
                    return self.evaluationFunction(state)

        # Return the next action which has the max score.
        next_act = ''
        maxscore = float("-inf")
        for action in gameState.getLegalActions(0):
            tmpscore = expectimax(gameState.generateSuccessor(0, action), 1, 1)
            if tmpscore > maxscore:
                maxscore = tmpscore
                next_act = action
        return next_act


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    pacman2food = sum([manhattanDistance(pacman_pos, food_pos) for food_pos in currentGameState.getFood()])
    pacman2ghost = 0
    ghostscaretime = 0
    for ghost in currentGameState.getGhostStates():
        pacman2ghost += manhattanDistance(pacman_pos, ghost.getPosition())
        ghostscaretime += ghost.scaredTimer
    score = currentGameState.getScore() + 1/pacman2food + ghostscaretime
    if ghostscaretime > 0:
        score -= pacman2ghost
    else:
        score += pacman2ghost
    return score
# Abbreviation
better = betterEvaluationFunction
