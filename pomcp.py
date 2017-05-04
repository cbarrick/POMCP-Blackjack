
import numpy as np
from blackjack import Agent, Action, DealerAgent
import random
import math


class Particle:
    def __init__(self, obs, s):
        self.obs = obs
        self.s = s

    @classmethod
    def from_obs(cls, obs):
        return cls(obs, obs.sample_state())

    @classmethod
    def from_s(cls, s):
        return cls(s.get_observation(), s)



class POMCP(Agent):
    '''POMCP algorithm

    SearchTree = class for maintaining search
    '''

    def __init__(self,
                 discount=0.8,
                 depth=0,
                 epsilon=1e-7,
                 explore=7,
                 n_particles=128,
                 reinvigoration=16):

        self.discount = discount
        self.depth = depth
        self.epsilon = epsilon
        self.explore = explore
        self.n_particles = n_particles
        self.reinvigoration = reinvigoration
        self.rollout_policy = DealerAgent()


    def __str__(self):
        return "POMCP"


    def policy(self, obs, ctx):
        tree = ctx.get('pomcp_root')

        at_root = tree is None

        if at_root:
            tree = SearchTree()
            ctx['pomcp_root'] = tree
            tree.particles = [Particle.from_obs(obs) for _ in range(self.n_particles)]
        else:
            particles = [part for part in tree.particles if part.obs == obs]
            for _ in range(self.reinvigoration):
                particles.append(Particle.from_obs(obs))
            tree.particles = particles

        for i in range(self.n_particles):
            part = random.sample(tree.particles, 1)[0]
            self.simulate(part, tree, 0)

        actions = obs.actions()
        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value)
        ctx['pomcp_root'] = child
        return child.action

    def simulate(self, part, tree, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(tree.children) == 0:

            tree.expand()
            return self.rollout(part, depth)
        actions = part.s.actions()
        if len(actions) is 0:
            return 0

        children = filter(lambda child: child.action in actions, tree.children)
        child = max(children, key=lambda child: child.value + self.explore * tree.ucb(child))
        action = child.action

        new_s = part.s.sample(action)
        new_part = Particle.from_s(new_s)
        reward = new_s.score() + self.discount * self.simulate(new_part, child, depth + 1)
        tree.particles.append(new_part)

        tree.visit += 1
        child.visit += 1
        child.value += (reward - child.value) / child.visit
        return reward

    def rollout(self, part, depth):
        if self.discount**depth < self.epsilon:
            return 0
        if len(part.s.actions()) == 0:
            return 0
        action = self.rollout_policy.policy(part.obs, {})
        new_s = part.s.sample(action)
        new_part = Particle.from_s(new_s)
        return new_s.score() + self.discount * self.rollout(new_part, depth + 1)


class SearchTree:
    def __init__(self, particles=[], action=None, visit=1, value=0):
        self.particles = particles
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
