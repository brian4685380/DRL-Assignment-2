import numpy as np
import random
def reverse(mat):
    return np.fliplr(mat)  # flip matrix left-right

def transpose(mat):
    return mat.T  # simple transpose

def cover_up(mat):
    new = np.zeros_like(mat)
    done = False
    for i in range(4):
        filtered = mat[i][mat[i] != 0]  # remove zeros
        new[i, :len(filtered)] = filtered
        if not np.array_equal(new[i], mat[i]):
            done = True
    return new, done

def merge(mat):
    done = False
    addPoints = 0
    for i in range(4):
        for j in range(3):
            if mat[i, j] == mat[i, j+1] and mat[i, j] != 0:
                mat[i, j] *= 2
                addPoints += mat[i, j]
                mat[i, j+1] = 0
                done = True
    return mat, done, addPoints

import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {key: "#776e65" if key in [2, 4] else "#f9f6f2" for key in COLOR_MAP if key != 0}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board
    
    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:    
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

            
    
    def move_left(self):
        moved = False
        game = self.board.copy()
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        add_points = temp[2]
        if not np.array_equal(game, self.board):
            moved = True
            self.score += add_points
            self.board = game
        return moved

    def move_right(self):
        moved = False
        game = self.board.copy()
        game = reverse(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        add_points = temp[2]
        game = reverse(game)
        if not np.array_equal(game, self.board):
            moved = True
            self.score += add_points
            self.board = game
        return moved
    def move_up(self):
        moved = False
        game = self.board.copy()
        game = transpose(game)
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        add_points = temp[2]
        game = transpose(game)
        if not np.array_equal(game, self.board):
            moved = True
            self.score += add_points
            self.board = game
        return moved
    def move_down(self):
        moved = False
        game = self.board.copy()
        game = reverse(transpose(game))
        game, done = cover_up(game)
        temp = merge(game)
        game = temp[0]
        done = done or temp[1]
        game = cover_up(game)[0]
        add_points = temp[2]
        game = transpose(reverse(game))
        if not np.array_equal(game, self.board):
            moved = True
            self.score += add_points
            self.board = game
        return moved
    
    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, spawn_tile=True):    
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False
        self.last_move_valid = moved

        # after_state = copy.deepcopy(self.board)
        if spawn_tile and moved:
        # if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def render_frame(self, action=None):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=8, fontweight='bold', color=text_color)

        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title, fontsize=8)
        plt.gca().invert_yaxis()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return Image.fromarray(img)

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:
            moved = self.simulate_move(temp_board, transpose, transpose_after=True)
        elif action == 1:
            moved = self.simulate_move(temp_board, lambda x: reverse(transpose(x)), transpose_after=True, reverse_after=True)
        elif action == 2:
            moved = self.simulate_move(temp_board, lambda x: x)
        elif action == 3:
            moved = self.simulate_move(temp_board, reverse, reverse_after=True)
        else:
            raise ValueError("Invalid action")

        return moved
    def simulate_move(self, board, transform, reverse_after=False, transpose_after=False):
        """Simulate a move safely on a board copy without touching self.board"""
        game = transform(board.copy())
        new_board, done = cover_up(game)
        temp = merge(new_board)
        new_board = temp[0]
        done = done or temp[1]
        new_board, _ = cover_up(new_board)

        if reverse_after:
            new_board = reverse(new_board)
        if transpose_after:
            new_board = transpose(new_board)

        return not np.array_equal(board, new_board)