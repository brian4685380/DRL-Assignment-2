from collections import defaultdict
import math
import numpy as np

def rot90(pattern):
    return tuple((y, 3 - x) for x, y in pattern)
def rot180(pattern):
    return tuple((3 - x, 3 - y) for x, y in pattern)
def rot270(pattern):
    return tuple((3 - y, x) for x, y in pattern)
def flip(pattern):
    return tuple((x, 3 - y) for x, y in pattern)
def flip_rot90(pattern):
    return tuple((3 - y, 3 - x) for x, y in pattern)
def flip_rot180(pattern):
    return tuple((3 - x, y) for x, y in pattern)
def flip_rot270(pattern):
    return tuple((y, x) for x, y in pattern)

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        sym = []
        sym.append(pattern)
        sym.append(rot90(pattern))
        sym.append(rot180(pattern))
        sym.append(rot270(pattern))
        sym.append(flip(pattern))
        sym.append(flip_rot90(pattern))
        sym.append(flip_rot180(pattern))
        sym.append(flip_rot270(pattern))
        return sym
    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)
    def value(self, board):
        values = []
        for i in range(len(self.patterns)):
            pattern_values = []
            for sym in self.symmetry_patterns[i]:
                index = tuple(self.get_feature(board, sym))
                pattern_values.append(self.weights[i][index])
            values.append(np.mean(pattern_values)) 
        return sum(values)
    def update(self, board, delta, alpha):
        for i in range(len(self.patterns)):
            for p in self.symmetry_patterns[i]:
                feature = self.get_feature(board, p)
                self.weights[i][feature] += alpha * delta