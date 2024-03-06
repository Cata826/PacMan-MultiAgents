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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        foodlist = newFood.asList()
        """A list with the positions of all ghosts"""
        position_ghosts = [ghost for ghost in successorGameState.getGhostPositions()]
        """A list with distances from the position of pacman and each food"""
        distances = [manhattanDistance(newPos, food) for food in foodlist]

        """If there is no food then return infinity"""
        if len(distances) == 0:
            return float('inf')
        extrascore = 0.0

        """The move is good if the distance from the nearest food has decreased"""
        min_distance = min(distances)
        extrascore = 1 / min_distance * 10

        for index, ghost in enumerate(position_ghosts):
            if newScaredTimes[index] > 0:
                """If we have eaten a capsule then we don't care about the ghost being near us"""
                continue
            else:
                """If the distance from the ghost is lower than 2 then return -infinity to avoid the ghost"""
                if manhattanDistance(ghost, newPos) < 2:
                    extrascore = float('-inf')

        return successorGameState.getScore() + extrascore


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        scores = [self.minimaxValue(gameState.generateSuccessor(0, action), 1, 1)
                  for action in legalActions]
        bestAction = max(zip(scores, legalActions))[1]
        return bestAction

    def minimaxValue(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Returns the minimax value of a gameState.
        """
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return max(self.minimaxValue(gameState.generateSuccessor(0, action),
                                         depth, 1) for action in gameState.getLegalActions(0))
        else:
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgentIndex == 0:
                depth += 1
            return min(self.minimaxValue(gameState.generateSuccessor(agentIndex, action),
                                         depth, nextAgentIndex) for action in gameState.getLegalActions(agentIndex))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """We return the action"""
        """
            alpha = -infinity
            beta = infinity
        """
        return self.maximize(gameState, (float('-inf'),), (float('inf'),), 0, 0, )[1]

    def alphabeta_pruning(self, state: GameState, alpha, beta, agent_index, depth):
        """If we reached the depth that we want to search or it's a win state or it's a lose state then we evaluate this state"""
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        """If it's pacman(index = 0) then we call maximize"""
        if agent_index == 0:
            return self.maximize(state, alpha, beta, agent_index, depth)[0]
        else:
            return self.minimize(state, alpha, beta, agent_index, depth)[0]

    def maximize(self, state: GameState, alpha, beta, agent_index, depth):
        maxValue = (float('-inf'),)
        legal_moves = state.getLegalActions(agent_index)

        for action in legal_moves:
            evalue = (
            self.alphabeta_pruning(state.generateSuccessor(agent_index, action), alpha, beta, agent_index + 1, depth),
            action)

            maxValue = self.max_eval(evalue, maxValue)
            alpha = self.max_eval(alpha, evalue)
            """If alpha is greater than beta then we prune"""
            if beta[0] < alpha[0]:
                break

        return maxValue

    def minimize(self, state: GameState, alpha, beta, agent_index, depth):
        minValue = (float('inf'),)
        legal_moves = state.getLegalActions(agent_index)

        for action in legal_moves:
            """If we have reached to the last ghost then we have to increase the depth and call minimax with 0 index"""
            if agent_index == state.getNumAgents() - 1:
                agent_index = -1
                depth += 1
            evalue = (
            self.alphabeta_pruning(state.generateSuccessor(agent_index, action), alpha, beta, agent_index + 1, depth),
            action)

            minValue = self.min_eval(evalue, minValue)
            beta = self.min_eval(beta, evalue)
            """If alpha is greater than beta then we prune"""
            if beta[0] < alpha[0]:
                break

        return minValue

    def max_eval(self, evaluation1, evaluation2):
        if evaluation1[0] > evaluation2[0]:
            return evaluation1
        else:
            return evaluation2

    def min_eval(self, evaluation1, evaluation2):
        if evaluation1[0] < evaluation2[0]:
            return evaluation1
        else:
            return evaluation2

class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legalActions = gameState.getLegalActions(0)  # Pacman's legal actions
        expectedValues = [self.expectimaxValue(gameState.generateSuccessor(0, action), 1, 1)
                          for action in legalActions]
        bestAction = max(zip(expectedValues, legalActions))[1]
        return bestAction

    def expectimaxValue(self, gameState: GameState, depth: int, agentIndex: int):
        """
        Returns the expectimax value of a gameState.
        """
        if depth > self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return max(self.expectimaxValue(gameState.generateSuccessor(0, action),
                                            depth, 1) for action in gameState.getLegalActions(0))
        else:  # Ghosts' turn
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgentIndex == 0:
                depth += 1
            ghostActions = gameState.getLegalActions(agentIndex)
            prob = 1.0 / len(ghostActions)
            return sum(prob * self.expectimaxValue(gameState.generateSuccessor(agentIndex, action),
                                                   depth, nextAgentIndex) for action in ghostActions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()


    score = currentGameState.getScore()

    minGhostDistance = min(manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates)

    remainingFood = len(foodList)
    evaluation = score - 2 * minGhostDistance - 10 * remainingFood

    return evaluation

# Abbreviation
better = betterEvaluationFunction
