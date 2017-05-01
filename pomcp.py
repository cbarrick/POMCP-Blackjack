import numpy as np
from blackjack import Agent, Action, DealerAgent
import random
import math


class POMCP(Agent):
    '''POMCP algorithm

    SearchTree = class for maintaining search
    '''

    def __init__(self, discount=0.9, depth=0, epsilon=1e-7, explore=6, n_particles=100):
        self.discount = discount
        self.depth = depth
        self.epsilon = epsilon
        self.explore = explore
        self.n_particles = n_particles
        self.rollout_policy = DealerAgent()

    def __str__(self):
        return "POMCP"

    def policy(self, obs, ctx):
        tree = ctx.get('pomcp_root')
        empty = tree is None
        if empty:
            tree = SearchTree()
            ctx['pomcp_root'] = tree

        for i in range(self.n_particles):
            if empty:
                s = obs.sample_state()
            else:
                s = random.sample(tree.belief, 1)[0]
            self.simulate(s, tree, 0)

        actions = obs.actions()
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value)
        ctx['pomcp_root'] = child
        return child.action

    def simulate(self, s, tree, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(tree.children) == 0:
            tree.expand()
            return self.rollout(s, depth)
        actions = s.actions()
        if len(actions) is 0:
            return 0
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        action = child.action

        new_s = s.sample(action)
        reward = new_s.score() + self.discount * self.simulate(new_s, child, depth + 1)
        tree.belief.append(s)
        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, s, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(s.actions()) == 0:
            return 0
        action = self.rollout_policy.policy(s, {})
        new_s = s.sample(action)
        return new_s.score() + self.discount * self.rollout(new_s, depth + 1)


class SearchTree:
    def __init__(self, belief=[], action=None, visit=1, value=0):
        self.belief = belief
        self.visit = visit
        self.value = value
        self.action = action
        self.children = set()

    def expand(self):
        for a in Action:
            self.children.add(SearchTree(action=a))

    def ucb(self, child):
        log = math.log(self.visit, len(self.children))
        div = log / child.visit
        return math.sqrt(div)
