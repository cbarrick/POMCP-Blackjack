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

    def policy(self, obs, ctx):
        tree = ctx.get('pomcp_root')
        if tree is None:
            tree = SearchTree(belief={obs.sample_belief()})
            ctx['pomcp_root'] = tree
        if len(tree.belief) == 0:
            tree.belief.add(obs.sample_belief())
        s = random.sample(tree.belief, 1)[0]
        self.simulate(obs, s, tree, 0)
        actions = obs.actions()

        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value)
        ctx['pomcp_root'] = child
        return child.action

    def simulate(self, obs, s, tree, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(tree.children) == 0:
            tree.expand(obs, s)
            return self.rollout(obs, s, depth)
        actions = obs.actions()
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        action = child.action

        new_obs = obs.sample(s, action)
        new_s = new_obs.sample_belief()
        reward = new_obs.score() + self.discount * simulate(new_obs, new_s, child, depth + 1)
        tree.belief.add(s)
        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, obs, s, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(obs.actions()) == 0:
            return 0
        action = self.rollout_policy(obs, {})
        new_obs = obs.sample(s, action)
        new_s = new_obs.sample_belief()
        return new_obs.score() + self.discount * self.rollout(new_obs, new_s, depth + 1)


class SearchTree:
    def __init__(self, belief=set(), action=None, visit=0, value=0):
        self.belief = belief
        self.visit = visit
        self.value = value
        self.action = action
        self.children = set()

    def expand(self, obs, s):
        for a in Action:
            self.children.add(SearchTree(action=a))

    def ucb(self, child):
        return math.sqrt(math.log(self.visit, len(children)) / child.visit)
