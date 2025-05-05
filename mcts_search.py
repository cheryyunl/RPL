import math
import random
from model import State

class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.children = {}  # dict(action -> MCTSNode)
        self.visit_count = 0
        self.value_sum = 0.0
        self.parent = None
        self.action_from_parent = None

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

def ucb_score(parent, child, c_puct=1.0):
    if child.visit_count == 0:
        return float('inf')
    return (child.value
            + c_puct * math.sqrt(math.log(parent.visit_count) / (child.visit_count)))

def select_child(node, c_puct=1.0):
    best_score = -float('inf')
    best_action = None
    best_child = None
    for action, child in node.children.items():
        score = ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child

def expand(node, vlm, generation_config, top_k=3):
    if node.state.is_terminal:
        return
    actions_probs = vlm.propose_actions(node.state, generation_config, top_k)
    for action, prob in actions_probs:
        if action not in node.children:
            next_state = vlm.transition(node.state, action)
            child_node = MCTSNode(next_state)
            child_node.parent = node
            child_node.action_from_parent = action
            node.children[action] = child_node

def simulate(state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, rollout_limit=10):
    temp_state = state.copy()
    steps = 0
    while not temp_state.is_terminal and steps < rollout_limit:
        actions_probs = vlm.propose_actions(temp_state, generation_config, top_k=1)
        action, prob = random.choice(actions_probs)
        temp_state = vlm.transition(temp_state, action)
        steps += 1

    return vlm.evaluate_terminal_state(temp_state, eval_llm, eval_llm_tokenizer, question, answer), temp_state

def backpropagate(node, reward):
    cur = node
    while cur is not None:
        cur.visit_count += 1
        cur.value_sum += reward
        cur = cur.parent

def mcts_search(root_state, vlm, eval_llm, eval_llm_tokenizer, question, answer, generation_config, n_iterations,
                c_puct=1.0, top_k=3):
    root_node = MCTSNode(root_state)
    solution = None
    # Store diverse reasoning paths
    diverse_solutions = []

    for iter in range(n_iterations):
        node = root_node
        while not node.state.is_terminal and len(node.children) > 0:
            _, child = select_child(node, c_puct)
            node = child

        if not node.state.is_terminal:
            expand(node, vlm, generation_config, top_k=top_k)
            if len(node.children) > 0:
                action = random.choice(list(node.children.keys()))
                node = node.children[action]

        reward, simulate_state = simulate(node.state, vlm, eval_llm, eval_llm_tokenizer, question, answer,
                                          generation_config, rollout_limit=10)
        
        # Store diverse solutions for preference learning
        if reward > 0:
            solution_text = "".join(simulate_state.solution_steps)
            # Check if this solution is significantly different from existing ones
            is_diverse = True
            for existing_sol in diverse_solutions:
                if similarity_score(solution_text, existing_sol["text"]) > 0.8:  # Simple similarity threshold
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_solutions.append({
                    "text": solution_text,
                    "reward": reward,
                    "state": simulate_state
                })
                
            # Keep the first solution found
            if solution is None:
                solution = simulate_state

        backpropagate(node, reward)

    best_path = []
    current = root_node
    while not current.state.is_terminal and len(current.children) > 0:
        best_child = max(current.children.values(), key=lambda c: c.visit_count)
        best_path.append(best_child.action_from_parent.text)
        current = best_child
        
    return root_node, best_path, solution, diverse_solutions, iter

# Simple text similarity function (could be improved with better NLP techniques)
def similarity_score(text1, text2):
    # Very simple similarity based on word overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if len(words1) == 0 or len(words2) == 0:
        return 0
    overlap = len(words1.intersection(words2))
    return overlap / max(len(words1), len(words2))