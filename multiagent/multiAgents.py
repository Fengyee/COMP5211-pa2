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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        
        ghostDistance = manhattanDistance(newPos, newGhostStates[0].getPosition())
        if ghostDistance > 0:
            score = score - 10 / ghostDistance

        foodDistances = []
        for food in newFood.asList():
            foodDistances.append(manhattanDistance(newPos, food))
        if len(foodDistances) > 0:
            score = score + 10 / min(foodDistances)
        
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        def _minimax_(state, depth, agent):
            if agent == state.getNumAgents():
                return _minimax_(state, depth + 1, 0)
            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state)
            
            successorScores = []
            for action in state.getLegalActions(agent):
                successorScores.append(_minimax_(state.generateSuccessor(agent, action), depth, agent + 1))

            if agent % state.getNumAgents() == 0:
                return max(successorScores)
            else:
                return min(successorScores)

        return max(gameState.getLegalActions(0), key=lambda x: _minimax_(gameState.generateSuccessor(0, x), 0, 1))

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def _exploreTree(state, depth, agent, alpha, beta):
            if agent == state.getNumAgents():
                depth += 1
                agent = 0
            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state), None
            if agent % state.getNumAgents() == 0:
                return evaluation(state, depth, agent, alpha, beta, float('-inf'), True)
            else:
                return evaluation(state, depth, agent, alpha, beta, float('inf'), False)                

        def evaluation(state, depth, agent, alpha, beta, value, isMaxNode):
            selectedAction = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                v, _ = _exploreTree(successor, depth, agent + 1, alpha, beta)
                value, selectedAction = (max if isMaxNode else min)((value, selectedAction), (v, action))

                if isMaxNode:
                    if value > beta:
                        return value, selectedAction
                    alpha = max(alpha, value)
                else:
                    if value < alpha:
                        return value, selectedAction
                    beta = min(beta, value)

            return value, selectedAction
    
        _, action = _exploreTree(gameState, 0, 0, float('-inf'), float('inf'))
        return action

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
        def _exploreTree(state, depth, agent):
            if agent == state.getNumAgents():  # is pacman
                return _exploreTree(state, depth + 1, 0)  # start next depth

            if state.isWin() or state.isLose() or depth == self.depth or state.getLegalActions(agent) == 0:
                return self.evaluationFunction(state)

            successorScores = []
            for action in state.getLegalActions(agent):
                successorScores.append(_exploreTree(state.generateSuccessor(agent, action), depth, agent + 1))

            if agent % state.getNumAgents() == 0:
                return max(successorScores)
            else:
                # return min(successorScores)
                return random.choice(successorScores)

        # return the best of pacman's possible moves
        return max(gameState.getLegalActions(0), key = lambda x: _exploreTree(gameState.generateSuccessor(0, x), 0, 1))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    for ghost in ghostStates:
        distance = manhattanDistance(pos, ghostStates[0].getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:
                score += 100.0 / distance
            else:
                score -= 45.0 / distance

    distancesToFood = [manhattanDistance(pos, f) for f in food.asList()]
    if len(distancesToFood) > 0:
        score += 10.0 / min(distancesToFood)
    return score

# Abbreviation
better = betterEvaluationFunction

