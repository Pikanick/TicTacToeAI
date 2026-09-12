"""
Headless regression tests for games.py (assignment 2: adversarial search).

These don't need the tkinter GUI (tic-tac-toe.py) at all -- they exercise
the search algorithms and evaluation function directly, which is what
actually matters for grading/correctness. Run with:

    python3 -m unittest test_games.py -v

or simply:

    python3 test_games.py
"""
import time
import unittest

from games import (
    TicTacToe, gen_state,
    minmax, minmax_cutoff, alpha_beta, alpha_beta_cutoff,
    random_player, query_player, SearchTimeout,
)


class EvalFunctionTests(unittest.TestCase):
    """eval1() is supposed to give the highest value to board configurations
    with the most potential for a win. Before the fix it always returned 0
    for every state (see games.py history), which meant minmax_cutoff and
    alpha_beta_cutoff never actually looked at the board at all."""

    def test_recognizes_immediate_win_for_x(self):
        game = TicTacToe(3, 3, -1)
        # X has two in the (1,*) column; (1,3) completes it.
        state = gen_state(to_move='X', x_positions=[(1, 1), (1, 2)], o_positions=[(2, 1)])
        self.assertGreater(game.eval1(state), 0)

    def test_recognizes_immediate_threat_from_o(self):
        game = TicTacToe(3, 3, -1)
        # O has two in the (1,*) column; (1,3) completes it for O.
        state = gen_state(to_move='X', x_positions=[(2, 1)], o_positions=[(1, 1), (1, 2)])
        self.assertLess(game.eval1(state), 0)

    def test_no_immediate_threats_is_neutral_ish(self):
        game = TicTacToe(3, 3, -1)
        state = gen_state(to_move='X', x_positions=[], o_positions=[])
        # Empty board: no immediate win available to either side.
        self.assertEqual(game.eval1(state), 0)


class CutoffSearchTests(unittest.TestCase):
    """These exercise minmax_cutoff/alpha_beta_cutoff end-to-end: with the
    old always-zero eval1, both of these would pick moves in a fixed,
    board-independent order (max() over a list of equal keys just returns
    the first one), so they would *not* reliably take a free win."""

    def test_minmax_cutoff_takes_the_winning_move(self):
        game = TicTacToe(3, 3, -1)
        game.d = 1
        state = gen_state(to_move='X', x_positions=[(1, 1), (1, 2)], o_positions=[(2, 1)])
        self.assertEqual(minmax_cutoff(game, state), (1, 3))

    def test_alpha_beta_cutoff_takes_the_winning_move(self):
        game = TicTacToe(3, 3, -1)
        game.d = 1
        state = gen_state(to_move='X', x_positions=[(1, 1), (1, 2)], o_positions=[(2, 1)])
        self.assertEqual(alpha_beta_cutoff(game, state), (1, 3))

    def test_deadline_raises_search_timeout(self):
        game = TicTacToe(4, 4, -1)
        game.d = 50  # deep enough that it won't finish before the deadline
        state = gen_state(to_move='X', x_positions=[], o_positions=[], h=4, v=4)
        already_past = time.perf_counter() - 1
        with self.assertRaises(SearchTimeout):
            minmax_cutoff(game, state, deadline=already_past)
        with self.assertRaises(SearchTimeout):
            alpha_beta_cutoff(game, state, deadline=already_past)


class FullStrengthSearchNeverLosesTests(unittest.TestCase):
    """Tic-tac-toe is a solved game: optimal play can always force at least
    a draw, for either side. This is a strong, deterministic invariant we
    can use to sanity-check the game engine (result/terminal_test/utility)
    regardless of eval1, by using the uncapped searches (game.timer = -1)
    against a random opponent, many times."""

    TRIALS = 15

    def test_alpha_beta_as_x_never_loses_to_random(self):
        for _ in range(self.TRIALS):
            game = TicTacToe(3, 3, -1)

            def ai(game, state):
                game.d = -1
                return alpha_beta(game, state)

            result = game.play_game(ai, random_player)
            self.assertGreaterEqual(result, 0, "optimal X should never lose")

    def test_minmax_as_x_never_loses_to_random(self):
        for _ in range(self.TRIALS):
            game = TicTacToe(3, 3, -1)

            def ai(game, state):
                game.d = -1
                return minmax(game, state)

            result = game.play_game(ai, random_player)
            self.assertGreaterEqual(result, 0, "optimal X should never lose")

    def test_alpha_beta_as_o_never_loses_to_random(self):
        for _ in range(self.TRIALS):
            game = TicTacToe(3, 3, -1)

            def ai(game, state):
                game.d = -1
                return alpha_beta(game, state)

            # utility() in play_game is always reported from X's perspective,
            # so when the optimal player is O (second), X should never *win*.
            result = game.play_game(random_player, ai)
            self.assertLessEqual(result, 0, "optimal O should never lose")


class QueryPlayerSafetyTests(unittest.TestCase):
    """query_player() used to run raw input() text through eval(), which
    executes arbitrary Python. It now uses ast.literal_eval, which only
    parses Python literals."""

    def test_parses_move_tuple(self, monkeypatch=None):
        game = TicTacToe(3, 3, -1)
        state = gen_state(to_move='X', x_positions=[], o_positions=[])
        import builtins
        original_input = builtins.input
        builtins.input = lambda prompt='': "(1, 1)"
        try:
            move = query_player(game, state)
        finally:
            builtins.input = original_input
        self.assertEqual(move, (1, 1))

    def test_does_not_execute_arbitrary_code(self):
        game = TicTacToe(3, 3, -1)
        state = gen_state(to_move='X', x_positions=[], o_positions=[])
        import builtins
        original_input = builtins.input
        marker = {"ran": False}
        builtins.input = lambda prompt='': "__import__('builtins')"
        try:
            move = query_player(game, state)
        finally:
            builtins.input = original_input
        # literal_eval rejects this (it isn't a literal), so it falls back
        # to treating it as a plain (invalid) move string rather than
        # executing __import__.
        self.assertFalse(marker["ran"])
        self.assertEqual(move, "__import__('builtins')")


if __name__ == '__main__':
    unittest.main()
