# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import copy
import random
import numpy as np
from collections import defaultdict
from Game2048Env import Game2048Env
from NTupleApproximator import NTupleApproximator
import sys
sys.modules['__main__'].NTupleApproximator = NTupleApproximator

patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(1, 0), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2)],
]
_GLOBAL_APPROXIMATOR = NTupleApproximator(board_size=4, patterns=patterns)
with open('./best_avg_remix-td_approximator.pkl', 'rb') as f:
    _GLOBAL_APPROXIMATOR = pickle.load(f)

class stateNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.child_nodes = {}
        self.visits = 0
        self.value = 0
    def is_fully_expanded(self, legal_actions=None):
        if legal_actions is None:
            return True
        return all(a in self.child_nodes for a in legal_actions)
    def select_child(self, legal_actions, c):
        for a in legal_actions:
            if a not in self.child_nodes:
                return a
        max_value = float('-inf')
        best_action = None
        for action in legal_actions:
            child = self.child_nodes[action]
            if child.visits == 0 or self.visits == 0:
                return action
            value = child.value + c * np.sqrt(np.log(self.visits) / child.visits)
            if value > max_value:
                max_value = value
                best_action = action
        return best_action
    def add_child(self, action, afterstate_node):
        self.child_nodes[action] = afterstate_node
        
class afterStateNode:
    def __init__(self, state, parent=None, reward=0):
        self.state = state
        self.parent = parent
        self.child_nodes = {}
        self.visits = 0
        self.value = 0
        self.reward = reward
    def is_fully_expanded(self, empty_cells):
        return len(self.child_nodes) == len(empty_cells) * 2
    def select_child(self):
        placements = list(self.child_nodes.keys())
        
        positions = defaultdict(list)
        for row, col, value in placements:
            positions[(row, col)].append((value, (row, col, value)))
            
        chosen_pos = random.choice(list(positions.keys()))
        
        candidates = positions[chosen_pos]
        
        if len(candidates) == 1:
            return candidates[0][1]
        
        if random.random() < 0.9:
            placement = next((placement for value, placement in candidates if value == 2), None)
        else:
            placement = next((placement for value, placement in candidates if value == 4), None)
        return placement
    def add_child(self, postition, value, state_node):
        self.child_nodes[(postition[0], postition[1], value)] = state_node



class TD_MCTS:
    def __init__(self, approximator, iterations=50, exploration_constant=0, rollout_depth=0):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
    def create_env_from_state(self, state):
        env = Game2048Env()
        env.board = state.copy()
        return env
    def select_action(self, state, env):
        root = stateNode(state)
        legal_actions = [a for a in range(4) if env.is_move_legal(a)]

        if not legal_actions:
            return 0
        for _ in range(self.iterations):
            leaf_node, update_path = self.select(root, legal_actions)
            if leaf_node.visits > 0 or leaf_node == root:
                leaf_node = self.expand(leaf_node)
            value = self.simulate(leaf_node)
            self.backpropogate(leaf_node, value, update_path)
        return self.best_action(root, legal_actions)
        
    def select(self, node, legal_actions):
        update_path = []
        
        while True:
            # Handle stateNode (player's turn to select an action)
            if isinstance(node, stateNode):
                if legal_actions is None:
                    sim_env = self.create_env_from_state(node.state)
                    legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
                
                if not legal_actions:  # No legal moves
                    return node, update_path
                    
                if not node.is_fully_expanded(legal_actions):
                    # Return this node for expansion if not all actions are expanded
                    for action in legal_actions:
                        if action not in node.child_nodes:
                            return node, update_path
                
                # Select best child according to UCB
                action = node.select_child(legal_actions, self.c)
                update_path.append((node, action))
                node = node.child_nodes[action]
                legal_actions = None  # Reset for next node
            
            # Handle afterStateNode (environment's turn to place a tile)
            elif isinstance(node, afterStateNode):
                empty_cells = list(zip(*np.where(node.state == 0)))
                
                if not empty_cells:  # Terminal state with no empty cells
                    return node, update_path
                    
                if not node.is_fully_expanded(empty_cells):
                    # Return this node for expansion
                    return node, update_path
                
                # All empty cells have been expanded, select one according to policy
                # Use the existing select_child method which doesn't take parameters
                placement = node.select_child()
                update_path.append((node, placement))
                node = node.child_nodes[placement]
            
            # We've reached a leaf node that's not a stateNode or afterStateNode
            else:
                return node, update_path

    
    def expand(self, node):
        if isinstance(node, stateNode):
            sim_env = self.create_env_from_state(node.state)
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            for a in legal_actions:
                if a not in node.child_nodes:
                    sim_env = self.create_env_from_state(node.state)
                    score = sim_env.score
                    sim_env.step(a, spawn_tile=False)
                    new_score = sim_env.score
                    reward = new_score - score
                    afternode = afterStateNode(sim_env.board, node, reward)
                    node.add_child(a, afternode)
                
            for action in legal_actions:
                if action in node.child_nodes and node.child_nodes[action].visits == 0:
                    return node.child_nodes[action]
            
            if legal_actions and legal_actions[0] in node.child_nodes:
                return node.child_nodes[legal_actions[0]]
            return node
        
        elif isinstance(node, afterStateNode):
            empty_cells = list(zip(*np.where(node.state == 0)))
            if not empty_cells:
                return node
            for x,y in empty_cells:
                new_board = node.state.copy()
                new_board[x][y] = 2
                state_node = stateNode(new_board, node)
                node.add_child((x,y), 2, state_node)
            for x,y in empty_cells:
                new_board = node.state.copy()
                new_board[x][y] = 4
                state_node = stateNode(new_board, node)
                node.add_child((x,y), 4, state_node)
            if empty_cells:
                pos = random.choice(empty_cells)
                key = (pos[0], pos[1], 2 if random.random() < 0.9 else 4)
                if key not in node.child_nodes:
                    keys = list(node.child_nodes.keys())
                    if keys:
                        key = keys[0]
                    else:
                        return node
                return node.child_nodes[key]
        else:
            raise ValueError("Invalid node type")
        
    def simulate(self, node):
        state = node.state
        if isinstance(node, afterStateNode):
            return node.reward + self.approximator.value(state)
        elif isinstance(node, stateNode):
            sim_env = self.create_env_from_state(state)
            legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_actions:
                return 0
            max_value = float('-inf')
            for a in legal_actions:
                sim_env_copy = self.create_env_from_state(state)
                score = sim_env_copy.score
                sim_env_copy.step(a, spawn_tile=False)
                afterstate =  sim_env_copy.board
                new_score = sim_env_copy.score
                value = new_score - score + self.approximator.value(afterstate)
                max_value = max(max_value, value)
            return max_value
    def backpropogate(self, node, value, update_path):
        node.visits += 1
        node.value = value
        for parent, action in reversed(update_path):
            parent.visits += 1
            if isinstance(parent, stateNode):
                max_value = float('-inf')
                for _, child in parent.child_nodes.items():
                    max_value = max(max_value, child.value)
                if max_value != float('-inf'):
                    parent.value = max_value
            elif isinstance(parent, afterStateNode):
                child = parent.child_nodes[action]
                parent.value = child.value
    def best_action(self, root, legal_actions):
        best_value = float('-inf')
        best_action = None
        for action in legal_actions:
            if action in root.child_nodes and root.child_nodes[action].visits > 0:
                value = root.child_nodes[action].value
                if value > best_value:
                    best_value = value
                    best_action = action
        return best_action
        
    def _create_env(self, board):
        """Create a temporary environment with the given board state."""
        env = Game2048Env()
        env.board = board.copy()
        return env

td_mcts = TD_MCTS(approximator = _GLOBAL_APPROXIMATOR, iterations = 50, exploration_constant = 0, rollout_depth = 0)
env = Game2048Env()

def get_action(state, score):
    env.board = state.copy()
    env.score = score
    best_act = td_mcts.select_action(state = state, env = env)
    return best_act
    