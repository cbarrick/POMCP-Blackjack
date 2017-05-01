import numpy as np
from blackjack import Agent, Action, DealerAgent
import random
import math


class POMCP(Agent):
    '''POMCP algorithm

    SearchTree = class for maintaining search
    '''

    def __init__(self, discount=0.9, depth=0, epsilon=1e-7, explore=0):
        self.discount = discount
        self.depth = depth
        self.epsilon = epsilon
        self.explore = explore
        self.rollout_policy = DealerAgent()

    def __str__(self):
        return "POMCP"

    def policy(self, obs, ctx):
        for i in range(100):
            tree = ctx.get('pomcp_root')
            if tree is None:
                tree = SearchTree(belief={obs.sample_belief()})
                ctx['pomcp_root'] = tree
            if len(tree.belief) == 0:
                tree.belief.add(obs.sample_belief())
            bel = random.sample(tree.belief, 1)[0]
            self.simulate(bel, tree, 0)

        actions = obs.actions()
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value)
        ctx['pomcp_root'] = child
        return child.action

    def simulate(self, bel, tree, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(tree.children) == 0:
            tree.expand()
            return self.rollout(bel, depth)
        actions = bel.actions()
        if len(actions) is 0:
            return 0
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        action = child.action

        new_bel = bel.sample(action)
        reward = new_bel.score() + self.discount * self.simulate(new_bel, child, depth + 1)
        tree.belief.add(bel)
        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, bel, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(bel.actions()) == 0:
            return 0
        action = self.rollout_policy(bel, {})
        new_bel = bel.sample(action)
        return new_bel.score() + self.discount * self.rollout(new_bel, depth + 1)


class SearchTree:
    def __init__(self, belief=set(), action=None, visit=1, value=0):
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
