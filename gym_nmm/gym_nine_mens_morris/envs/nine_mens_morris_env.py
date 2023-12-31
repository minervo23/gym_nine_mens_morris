import gym
import numpy as np
from gym import spaces


# Definition der Pix-Klasse
class Pix:
    # Definition von S, W und B als Unterklasse
    class S:
        string = 'O' # Gibt Spielstein als String aus
        tup = (1, 0, 0) # Tupel repräsentiert Spielstein
        arr = np.array(tup) # Array, das ebenfalls den Spielstein repräsentiert

    class W:
        string = 'Weiss'
        tup = (0, 1, 0)
        arr = np.array(tup)
        idx = np.array([0, 2])

    class B:
        string = 'Schwarz'
        tup = (0, 0, 1)
        arr = np.array(tup)
        idx = np.array([1, 3])

    # Konvertierung von Tupel zu String und umgekehrt
    tup_to_str = {
        S.tup: '•',
        W.tup: W.string,
        B.tup: B.string
    }

    str_to_tup = {
        S.string: S.tup,
        W.string: W.tup,
        B.string: B.tup
    }

legal_moves = {

    # Alle Ecken
    # Format: (position): [oben, rechts, unten, links]
    # Wenn ein Zug nicht erlaubt ist, wird None angegeben    (0, 0, 0): [None, None, (0, 1, 1), (0, 1, 0)],
    (0, 0, 1): [(0, 1, 1), None, None, (0, 1, 2)],
    (0, 0, 2): [(0, 1, 3), (0, 1, 2), None, None],
    (0, 0, 3): [None, (0, 1, 0), (0, 1, 3), None],

    (1, 0, 0): [None, None, (1, 1, 1), (1, 1, 0)],
    (1, 0, 1): [(1, 1, 1), None, None, (1, 1, 2)],
    (1, 0, 2): [(1, 1, 3), (1, 1, 2), None, None],
    (1, 0, 3): [None, (1, 1, 0), (1, 1, 3), None],

    (2, 0, 0): [None, None, (2, 1, 1), (2, 1, 0)],
    (2, 0, 1): [(2, 1, 1), None, None, (2, 1, 2)],
    (2, 0, 2): [(2, 1, 3), (2, 1, 2), None, None],
    (2, 0, 3): [None, (2, 1, 0), (2, 1, 3), None],

    # All edges
    (0, 1, 0): [None, (0, 0, 0), (1, 1, 0), (0, 0, 3)],
    (0, 1, 1): [(0, 0, 0), None, (0, 0, 1), (1, 1, 1)],
    (0, 1, 2): [(1, 1, 2), (0, 0, 1), None, (0, 0, 2)],
    (0, 1, 3): [(0, 0, 3), (1, 1, 3), (0, 0, 2), None],

    (1, 1, 0): [(0, 1, 0), (1, 0, 0), (2, 1, 0), (1, 0, 3)],
    (1, 1, 1): [(1, 0, 1), (0, 1, 1), (1, 0, 1), (2, 1, 1)],
    (1, 1, 2): [(2, 1, 2), (1, 0, 1), (0, 1, 2), (1, 0, 2)],
    (1, 1, 3): [(1, 0, 3), (2, 1, 3), (1, 0, 2), (0, 1, 3)],

    (2, 1, 0): [(1, 1, 0), (2, 0, 0), None, (2, 0, 3)],
    (2, 1, 1): [(2, 0, 0), (1, 1, 1), (2, 0, 1), None],
    (2, 1, 2): [None, (2, 0, 1), (1, 1, 2), (2, 0, 2)],
    (2, 1, 3): [(2, 0, 3), None, (2, 0, 2), (1, 1, 3)],
}


class NineMensMorrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    #Ähnlich wie Info Types, die für die KI gesetzt werden können
    class InfoCode:
        normal = 0
        bad_action_position = 1
        bad_move = 2
        bad_kill_position = 3

    def __init__(self):
        """
        repräsentiert den Raum aller möglichen Aktionen, die der Agent in dieser Umgebung ausführen kann.
        Es gibt vier unterschiedliche Typen von Entscheidungen/Aktionen, die getroffen werden können.
        Die Zahlen (3, 2, 4, 4) bedeuten:
        Eine Entscheidung mit 3 Optionen.
        Eine Entscheidung mit 2 Optionen.
        Eine Entscheidung mit 4 Optionen.
        Noch eine Entscheidung mit 4 Optionen.
        """
        self.action_space = spaces.MultiDiscrete((3, 2, 4, 4))

        """
        repräsentiert den Raum aller möglichen Beobachtungen oder Zustände, die der Agent in dieser Umgebung sehen kann. 
        Es besteht aus zwei Hauptteilen.
        """
        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([3, 2, 4, 3]),
            spaces.MultiDiscrete([4, 9])
        ))

        self.board = None
        self.mens = None
        self.is_done = False  # True when episode is complete
        self._player = Pix.B
        self._opponents = {Pix.W: Pix.B, Pix.B: Pix.W}

    @property
    def player(self):
        return self._player

    @property
    def opponent(self):
        return self._opponents.get(self.player)

    @player.setter
    def player(self, player):
        if self.player == Pix.W or self.player == Pix.B:
            self._player = player
        else:
            raise Exception("Player must be either Pix.W or Pix.O")

# Führt einen Spielzug durch und gibt den neuen Zustand des Spiels, eine Belohnung, ob das Spiel beendet ist und zusätzliche Informationen zurück.
    def step(self, action):
        """
        :param action: (tuple)
        0: position: a 3,2,4 tuple
        1: move: a scalar
        2: kill_location: position tuple where the opponent's piece is removed.
        :return: state, reward, is_done, info
        """
        position, move, kill_location = action

        winner = self._winner()
        self.is_done = bool(winner)
        if self.is_done:
            return (self.board, self.mens), 0, self.is_done, {'code': self.InfoCode.bad_action_position,
                                                              'winner': winner.string}

        position = tuple(position)
        unused, killed = self.mens[self.player.idx]
        moved_position = self._get_moved_position(position, move)
        is_phase_1 = unused > 0

        is_illegal = self._is_action_illegal(position, moved_position, is_phase_1, kill_location)
        if is_illegal:
            return (self.board, self.mens), 0, self.is_done, {'code': is_illegal}

        # Update board
        old_state = np.array(self.board), np.array(self.mens)
        if is_phase_1:
            self.board[position] = self.player.arr
            self.mens[self.player.idx[0]] -= 1  # Unused will be reduced by 1
        else:
            self.board[position] = Pix.S.arr
            self.board[moved_position] = self.player.arr

        reward = 0

        has_killed = self._has_killed(position if is_phase_1 else moved_position, self.board)
        if has_killed:
            if kill_location is None:
                self.board, self.mens = old_state
                return (self.board, self.mens), reward, self.is_done, {'code': self.InfoCode.bad_kill_position}
            reward = 10
            self.mens[self.opponent.idx[1]] += 1
            self.board[tuple(kill_location)] = Pix.S.arr

        info = {'code': self.InfoCode.normal}

        winner = self._winner()
        self.is_done = bool(winner)
        if self.is_done:
            if self.player == winner:
                reward = 100
            info['winner'] = winner.string

        self.swap_players()

        return (self.board, self.mens), reward, self.is_done, info

    def reset(self):
        self.board, self.mens = self._get_empty_state()
        self.is_done = False

        self.swap_players()

        return self.board

    # Zeigt den aktuellen Zustand des Spiels an.
    def render(self, mode='human', close=False):
        print(f"Current Player: {self.player.string}")
        print(self.mens)
        self.print_board(self.board)

    def swap_players(self):
        self.player = self._opponents[self.player]

    def set_state(self, board_str, mens):
        self.reset()

        s = board_str

        self.board = np.array([
            # Outer layer
            [
                [Pix.str_to_tup[s[0][0]], Pix.str_to_tup[s[0][12]], Pix.str_to_tup[s[6][-1]], Pix.str_to_tup[s[6][0]]],
                [Pix.str_to_tup[s[3][0]], Pix.str_to_tup[s[0][6]], Pix.str_to_tup[s[3][-1]], Pix.str_to_tup[s[-1][6]]],
            ],

            # Middle layer
            [
                [Pix.str_to_tup[s[1][2]], Pix.str_to_tup[s[1][10]], Pix.str_to_tup[s[5][10]], Pix.str_to_tup[s[5][2]]],
                [Pix.str_to_tup[s[3][2]], Pix.str_to_tup[s[1][6]], Pix.str_to_tup[s[3][10]], Pix.str_to_tup[s[5][6]]],
            ],

            # Inner layer
            [
                [Pix.str_to_tup[s[2][4]], Pix.str_to_tup[s[2][8]], Pix.str_to_tup[s[4][8]], Pix.str_to_tup[s[4][4]]],
                [Pix.str_to_tup[s[3][4]], Pix.str_to_tup[s[2][6]], Pix.str_to_tup[s[3][8]], Pix.str_to_tup[s[4][6]]],
            ],
        ])

        self.mens = np.array(mens)
        self.is_done = bool(self._winner())

    def get_legal_actions(self):
        """
        1. If phase 1, return all positions where empty
        2. find all positions of player that are movable
        3. If any moved position kills the opponent, then get all opponent positions too
        """

        opponent_positions = np.transpose((self.board == self.opponent.arr).all(3).nonzero())
        all_legal_actions = []

        if self.mens[self.player.idx][0] > 0:
            open_positions = np.transpose((self.board == Pix.S.arr).all(3).nonzero())
            print(f"open_positions {open_positions}")
            for position in open_positions:
                position = tuple(position)
                print(f"position {position}")

                board = np.array(self.board)
                board[position] = self.player.arr
                print(f"position: {position}")
                print(f"board:\n{board}")
                has_killed = self._has_killed(position, board)
                if has_killed:
                    for opponent_position in opponent_positions:
                        all_legal_actions.append((position, None, opponent_position))
                else:
                    all_legal_actions.append((position, None, None))
            return all_legal_actions

        player_positions = np.transpose((self.board == self.player.arr).all(3).nonzero())

        for i in range(len(player_positions)):
            position = tuple(player_positions[i])
            for j in range(4):
                if legal_moves[position][j] is None:
                    continue
                move = tuple(legal_moves[position][j])
                if all(self.board[move] == Pix.S.arr):
                    # Now move to this position and check if killed, then gather all opponent positions
                    board = np.array(self.board)
                    board[move] = self.player.arr
                    board[position] = Pix.S.arr
                    has_killed = self._has_killed(move, board)
                    if has_killed:
                        for opponent_position in opponent_positions:
                            all_legal_actions.append((position, j, opponent_position))
                    else:
                        all_legal_actions.append((position, j, None))
        return all_legal_actions

    def is_phase_1(self):
        unused, _ = self.mens[self.player.idx]
        return unused > 0

    # ----- Private Methods ------

    def _is_action_illegal(self, position, moved_position, is_phase_1, kill_location=None):
        """
        Phase 1:
          - Position not empty
        Phase 2:
          - Position is non-player
          - Move is none
          - Moved position is out of bounds
          - Moved position is not empty
        """

        if is_phase_1:
            if any(self.board[position] != Pix.S.arr):
                return self.InfoCode.bad_action_position  # "During phase 1, the position must be empty."
        else:  # Phase 2
            if any(self.board[position] != self.player.arr):
                return self.InfoCode.bad_action_position  # "During phase 2, the position must be player's piece"
            if moved_position is None:  # Out of bounds
                return self.InfoCode.bad_move  # "Can't move the piece to that position."
            if any(self.board[moved_position] != Pix.S.arr):  # Is not empty
                return self.InfoCode.bad_move  # "The moved position must be empty."

        if kill_location is not None and any(self.board[tuple(kill_location)] != self.opponent.arr):
            return self.InfoCode.bad_kill_position  # "Invalid kill_location"

    def _winner(self):
        if self.mens[2] == 9:
            return Pix.B
        if self.mens[3] == 9:
            return Pix.W
        return False

    @staticmethod
    def _has_killed(recent_move, board):
        # Check all 4 edges of recently moved position.
        # If there's a 3 in a line, then remove desired piece.

        left, up, right, down = legal_moves[recent_move]
        left_2, up_2, right_2, down_2 = NineMensMorrisEnv._get_neighbours_level_2(recent_move)

        positions = [
            (left, right),
            (up, down),
            (left, left_2),
            (right_2, right),
            (up, up_2),
            (down, down_2)
        ]

        for pos1, pos2 in positions:
            if pos1 is not None and pos2 is not None:
                if all(board[pos1] == board[recent_move]) and all(board[pos2] == board[recent_move]):
                    return True

    @staticmethod
    def _get_neighbours_level_2(position):
        l, u, r, d = legal_moves[position]
        nones = [None, None, None, None]
        ll = legal_moves.get(l, nones)[0]
        uu = legal_moves.get(u, nones)[1]
        rr = legal_moves.get(r, nones)[2]
        dd = legal_moves.get(d, nones)[3]
        return ll, uu, rr, dd

    @staticmethod
    def _get_moved_position(position, move):
        """
        :param position: array of shape (3, 2, 4) -> position on the board.
        :param move: one of [0, 1, 2, 3]
        :return: position of the move.
        """

        if move is None:
            return

        return legal_moves[position][move]

    @staticmethod
    def _get_empty_state():
        board = np.zeros((3, 2, 4, 3), dtype=np.uint8)
        board[:, :, :, 0] = 1

        mens = np.array([9, 9, 0, 0])

        return board, mens

    @staticmethod
    def print_board(board):
        s = [
            (0, 0, 0), (0, 1, 1), (0, 0, 1),
            (1, 0, 0), (1, 1, 1), (1, 0, 1),
            (2, 0, 0), (2, 1, 1), (2, 0, 1),
            (0, 1, 0), (1, 1, 0), (2, 1, 0), (0, 1, 2), (1, 1, 2), (2, 1, 2),
            (2, 0, 3), (2, 1, 3), (2, 0, 3),
            (1, 0, 3), (1, 1, 3), (1, 0, 2),
            (0, 0, 3), (0, 1, 3), (0, 0, 2),
        ]
        s = [Pix.tup_to_str[tuple(board[x])] for x in s]
        string = f"""
{s[0]}-----{s[1]}-----{s[2]}
| {s[3]}---{s[4]}---{s[5]} |
| | {s[6]}-{s[7]}-{s[8]} | |
{s[9]}-{s[10]}-{s[11]}   {s[12]}-{s[13]}-{s[14]}
| | {s[15]}-{s[16]}-{s[17]} | |
| {s[18]}---{s[19]}---{s[20]} |
{s[21]}-----{s[22]}-----{s[23]}
"""
        print(string)

    @staticmethod
    def get_move_from_position(action_position, move_position):
        moves = legal_moves.get(tuple(action_position), [])
        for i in range(len(moves)):
            if moves[i] == tuple(move_position):
                return i
