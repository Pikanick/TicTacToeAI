"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
from collections import namedtuple
import numpy as np
import time

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')


def gen_state(move='(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(
        o_positions)  # unordered, basically values stored randomly in set
    moves = list(moves)  # values stored are indexed
    board = {}  # dictionary: associates key value pairs. Probably location on board with value
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:  # setting X and O values
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""
    player = game.to_move(state)  # sets the player who has the current turn given the current game state.

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state,
                                player)  # if we've arrived at a terminal state, return the utility (-1, 0 , or +1) of that game
        v = -np.inf  # setting value to negative infinity. np is just variable used from numpy library
        for a in game.actions(
                state):  # take all actions from this game state and return the value of the game that has the maximum across all possibile states from this game state
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)), default=None)


def minmax_cutoff(game, state):  # , end_time):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""
    player = game.to_move(state)  # determine current player

    def max_value(state, d):
        # print("Your code goes here -3pt")
        # print("MinMax with Cutoff (max_value) running at depth: ", d)
        if game.terminal_test(state) or d == 0:
            # print("Terminal state max at depth: ", d)
            return game.utility(state, player)  # game.utility(state, player)
        if d == game.d:
            return game.eval1(state)  # get an estimated value for non-terminal state
        v = -np.inf
        for move in game.actions(state):
            v = max(v, min_value(game.result(state, move), d + 1))  # increment d with each recursive call
            # print("Max d incremented to: ", d - 1)
        return v
        # return 0

    def min_value(state, d):
        # print("Your code goes here -2pt")
        # print("MinMax with Cutoff (min_value) running at depth: ", d)
        if game.terminal_test(state) or d == 0:
            # print("Terminal state min at depth: ", d)
            return game.utility(state, player)  # game.utility(state, player)
        if d == game.d:
            return game.eval1(state)
        v = np.inf
        for move in game.actions(state):
            v = min(v, max_value(game.result(state, move), d + 1))
            # print("Min d incremented to: ", d - 1)
        return v
        # return 0

    # Body of minmax_cutoff:
    # return max(game.actions(state), key=lambda a: min_value(game.result(state, a), 0), default=None)
    # best_score = -np.inf
    # best_move = None
    # for move in game.actions(state):  # check all possible actions
    #     v = min_value(game.result(state, move), 1)
    #     if v > best_score:  # Select action with highest value
    #         best_score = v
    #         best_move = move
    # return best_move

    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), 1), default=None)


    # if game.to_move(state) == player:  # consider whether its a maximizing or minimizing turn, if true its the current players turn (in the tree)
    #     best_move = max(game.actions(state), key=lambda a: min_value(game.result(state, a), game.d), default=None)  # Opponent's optimal response is being simulated in min_value
    #     print(f"Maximizing player chose move: {best_move}")
    #     return best_move
    # else:
    #     best_move = min(game.actions(state), key=lambda a: max_value(game.result(state, a), game.d), default=None)  # Player's optimal response is being simulated in max_value
    #     print(f"Minimizing player chose move: {best_move}")
    #     return best_move

# def minmax_cutoff(game, state):
#     """Search game to determine best action; use eval1 to evaluate non-terminal states at depth d_limit."""
#     player = game.to_move(state)
#     d_limit = game.d
#
#     def max_value(state, depth):
#         if game.terminal_test(state) or d_limit == 0:
#             return game.utility(state, player)
#         if depth == d_limit:
#             return game.eval1(state)
#         v = float('-inf')
#         for a in game.actions(state):
#             v = max(v, min_value(game.result(state, a), depth + 1))
#         return v
#
#     def min_value(state, depth):
#         if game.terminal_test(state) or d_limit == 0:
#             return game.utility(state, player)
#         if depth == d_limit:
#             return game.eval1(state)
#         v = float('inf')
#         for a in game.actions(state):
#             v = min(v, max_value(game.result(state, a), depth + 1))
#         return v
#
#     # Body of minmax_cutoff:
#     best_score = float('-inf')
#     best_action = None
#     for a in game.actions(state):
#         v = min_value(game.result(state, a), 1)
#         if v > best_score:
#             best_score = v
#             best_action = a
#     return best_action


# ______________________________________________________________________________


def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     this version searches all the way to the leaves."""
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        # print("Your code goes here -3pt")
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        # return 0

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        # print("Your code goes here -2pt")
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
        # return 0

    # Body of alpha_beta_search:
    alpha = -np.inf
    beta = np.inf
    best_move = None
    best_score = -np.inf
    # print("Your code goes here -10pt")
    # for move in game.actions(state):
    #     v = min_value(game.result(state, move), alpha, beta)
    #     if v > best_score:
    #         best_score = v
    #         best_move = move
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta), default=None)
    # return best_move


def alpha_beta_cutoff(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state) or depth == 0:
            return game.utility(state, player)
        # print("Your code goes here -3pt")
        if depth == game.d:
            return game.eval1(state)
        v = -np.inf
        for move in game.actions(state):
            v = max(v, min_value(game.result(state, move), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        # return 0

    def min_value(state, alpha, beta, depth):
        if game.terminal_test(state) or depth == 0:
            return game.utility(state, player)
        # print("Your code goes here -2pt")
        if depth == game.d:
            return game.eval1(state)
        v = np.inf
        for move in game.actions(state):
            v = min(v, max_value(game.result(state, move), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
        # return 0

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    alpha = -np.inf
    beta = np.inf
    # best_action = None
    # best_score = -np.inf
    # print("Your code goes here -10pt")
    # for move in game.actions(state):
    #     v = min_value(game.result(state, move), alpha, beta, 1)
    #     if v > best_score:
    #         best_score = v
    #         best_action = move
    # return best_action
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), alpha, beta, 1), default=None)
    # return best_action


# ______________________________________________________________________________
# Players for Games
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    """uses alphaBeta prunning with minmax, or with cutoff version, for AI player"""
    # print("Your code goes here -2pt")
    """Use a method to speed up at the start to avoid search down a long tree with not much outcome.
    Hint: for speedup use random_player for start of the game when you see search time is too long"""

    if game.timer < 0:
        game.d = -1
        return alpha_beta(game, state)

    start = time.perf_counter()
    end = start + game.timer
    """use the above timer to implement iterative deepening using alpha_beta_cutoff() version"""
    move = None
    # print("Your code goes here -10pt")
    depth = 1

    if game.timer == 0:
        game.d = depth
        print("timer out, selecting random")
        return random.choice(game.actions(state)) if game.actions(state) else None

    while time.perf_counter() < end:
        game.d = depth
        current_move = alpha_beta_cutoff(game, state)
        if time.perf_counter() < end:
            move = current_move  # Update the move only if there's still time left
        depth += 1

    print("iterative deepening to depth: ", game.d)
    return move


def minmax_player(game, state):
    """uses minmax or minmax with cutoff depth, for AI player"""
    # print("Your code goes here -3pt")
    """Use a method to speed up at the start to avoid search down a long tree with not much outcome.
    Hint:for speedup use random_player for start of the game when you see search time is too long"""

    if game.timer < 0:  # this is how the basic minmax function runs
        game.d = -1
        return minmax(game, state)

    start = time.perf_counter()
    # print("Start time: ", start)
    end = start + game.timer
    # print("End time: ", end)
    """use the above timer to implement iterative deepening using minmax_cutoff() version"""
    move = None
    # print("Your code goes here -10pt")
    depth = 1

    # second case, timer ran out, use random
    if game.timer == 0:
        game.d = depth
        print("timer out, selecting random")
        return random.choice(game.actions(state)) if game.actions(state) else None

    # third case, timer hasn't ran out, use minmax_cutoff
    while time.perf_counter() < end:
        game.d = depth
        current_move = minmax_cutoff(game, state)
        # print("time is now: ", time.perf_counter())
        # print("Start time: ", start)
        # print("End time: ", end)
        if time.perf_counter() < end:
            move = current_move  # Update the move only if there's still time left
        depth += 1

    print("iterative deepening to depth: ", game.d)
    return move


# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError  # state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1  # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size  # max depth possible is width X height of the board
        self.timer = t  # timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert (player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    # evaluation function, version 1
    def eval1(self, state):
        """design and implement evaluation function for state.
        Some ideas: 1-use the number of k-1 matches for X and O For this you can use function possibleKComplete().
            : 2- expand it for all k matches
            : 3- include double matches where one move can generate 2 matches.
            """

        """ computes number of (k-1) completed matches. This means number of row or columns or diagonals
        which include player position and in which k-1 spots are occupied by player.
        """

        # This needs to check if there are any wins for x or o horizontally, vertically or diagonnally, then give a value to that move/state if there is or isnt

        def possiblekComplete(move, board, player,
                              k):  # Basically checks if this move can complete any of the following lines on the board, rather than checking the whole board for each move.
            """if move can complete a line of count items, return 1 for 'X' player and -1 for 'O' player"""
            match = self.k_in_row(board, move, player, (0, 1), k)  # Vertical
            match = match + self.k_in_row(board, move, player, (1, 0), k)  # Horizontal
            match = match + self.k_in_row(board, move, player, (1, -1), k)  # Decreasing Diagonal
            match = match + self.k_in_row(board, move, player, (1, 1), k)  # Rising Diagonal
            # print("Match: ", match)
            return match

        # Maybe to accelerate, return 0 if number of pieces on board is less than half of board size:
        # if len(state.moves) <= self.k / 2:
        #     return 0
        # print("Your code goes here 15pt.")
        # print("Running Evaluation Function")
        # return 0

        # Check for immediate wins
        for move in state.moves:
            if self.k_in_row(state.board, move, 'X', (0, 1), self.k) or \
                    self.k_in_row(state.board, move, 'X', (1, 0), self.k) or \
                    self.k_in_row(state.board, move, 'X', (1, -1), self.k) or \
                    self.k_in_row(state.board, move, 'X', (1, 1), self.k):
                return +1  # 'X' wins
            if self.k_in_row(state.board, move, 'O', (0, 1), self.k) or \
                    self.k_in_row(state.board, move, 'O', (1, 0), self.k) or \
                    self.k_in_row(state.board, move, 'O', (1, -1), self.k) or \
                    self.k_in_row(state.board, move, 'O', (1, 1), self.k):
                return -1  # 'O' wins
            else:
                return 0  # tie

        score = 0

        # Calculate potential wins
        for move in state.moves:
            score += possiblekComplete(move, state.board, 'X', self.k - 1)
            score -= possiblekComplete(move, state.board, 'O', self.k - 1)

        # print("Score is: ", score)
        return score

    # @staticmethod
    def k_in_row(self, board, pos, player, dir, k):
        """Return true if there is a line of k cells in direction dir including position pos on board for player."""
        (delta_x, delta_y) = dir
        x, y = pos
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = pos
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= k
