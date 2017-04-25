'''A Blackjack simulator.

This module provides a simulator for the card game Blackjack. Its primary
purpose is to facilitate the comparison of differnt agent policies. We can
think of agents as belonging to one of two categories: *naïve* agents are
very general algorithms with no knowledge of the rules of Blackjack, while
*informed* agents are those with domain specific knowledge. An example of a
naïve agent would be Monte-Carlo Tree Search (and similarly POMCP) which can
search through arbitrary action spaces, while an example of an informed agent
would be a rule-based card counter.

We describing the module from a high level with regards to naïve agents, then
extend our discussion to the rules of Blackjack which may be (ab)used by
informed agents.

## Overview

The entry point into the module is through the `Simulator` class. Simulators
are responsible for maintaining the global state of the game and executing
agents. A simulator executes a dealer agent and one or more player agents over
an arbitrary number of rounds and provides statistics about the outcomes.

An agent is any callable object that takes two arguments, an `Observation` and
a context, and returns an `Action` indicating the next action to perform. If
you wish to implement your agent as a class, you should derive from the `Agent`
base class and implement your policy by overriding `Agent.policy(obs, ctx)`.

The context passed to an agent is an initially empty dict. The simulator never
mutates the context, and the same context is passed to the agent every time it
is called during the same round, allowing agents to persist round-level data.

`Observation` objects capture the knowledge of the world available to the agent
and allow the agent to sample from the space of possible future observations.
The `Observation.actions()` method returns a list of valid actions, and the
`Observation.sample(action)` method samples a new future observation given some
action. The method `Observation.score()` gives a score to the observation.
Naïve agents may sample and score an observation, but should otherwise treat
them as opaque.

`Action` objects represent the actions available to the agents. Naïve agents
will treat these as opaque vales.

## Rules of Blackjack

TODO: write this up
'''

from copy import copy, deepcopy
from enum import Enum
import random
import numpy as np
import logging

from agents import dealer_agent

logger = logging.getLogger(__name__)

class Action(Enum):
    '''The possible actions in a game of Blackjack.

    Enum:
        STAND: Perform no more actions this round.
        HIT: Draw another card.
        DOUBLE: Double your bet, hit, then stand.
    '''
    STAND = 1
    HIT = 2
    DOUBLE = 3
    # SPLIT = 4

class Shoe:
    '''A shuffled collection of French playing cards.'''

    def __init__(self, n_decks):
        '''Create a full Shoe with some number of decks of cards.'''
        assert n_decks > 0, '`n_decks` must be greater than 0.'
        self.n_decks = n_decks
        self.counts = {card_face: 4*n_decks for card_face in range(1,14)}

    def __len__(self):
        '''The length of a shoe is the number of cards.'''
        return sum(v for v in self.counts.values())

    def reshuffle(self):
        '''Refill and shuffle the shoe.'''
        self.__init__(self.n_decks)

    def draw(self):
        '''Draw a random card from the shoe.'''
        assert len(self) > 0, 'cannot draw from an empty shoe.'
        card = random.randrange(1,14)
        while self.counts[card] == 0:
            card = random.randrange(1,14)
        self.counts[card] -= 1
        return card

    def replace(self, card):
        '''Raplace a card back into the shoe at a random position.'''
        count = self.counts[card]
        assert count <= 4 * self.n_decks, f'cannot have more than {count} cards of value {card}.'
        self.counts[card] = count + 1

class State:
    '''A state of a round of Blackjack.'''

    def __init__(self, shoe, hands, dealer):
        '''Create a new state from a given shoe and hands dict.

        Args:
            shoe: The shoe to draw from when sampling the next state.
            hands: A dict mapping Agent instances to lists of cards.
            dealer: The agent to designate as the dealer.
        '''
        self.shoe = shoe
        self.hands = hands
        self.dealer = dealer
        self.stand = {agent:False for agent in hands.keys()}

    def __deepcopy__(self, memo):
        '''Returns a deepcopy of this State.'''
        s = copy(self)
        s.shoe = deepcopy(self.shoe, memo)
        s.hands = {agent:deepcopy(hand, memo) for agent, hand in self.hands.items()}
        return s

    def actions(self, agent):
        '''Returns the set of valid actions for the given agent.'''
        return tuple(Action) if not self.stand[agent] else []

    def sample(self, agent, action):
        '''Samples a new state from this one given an action taken by an agent.'''
        s = deepcopy(self)
        s.next(agent, action)
        return s

    def next(self, agent, action):
        '''Progress this state to the next by commiting an action as a player.'''
        assert isinstance(action, Action), 'action must be an instance of blackjack.Action'
        assert not self.stand[agent], 'cannot perform action after stand or bust'

        if action is Action.HIT or action is Action.DOUBLE:
            card = self.shoe.draw()
            self.hands[agent].append(card)
            self.stand[agent] = self.busted(agent)

        if action is Action.STAND or action is Action.DOUBLE:
            self.stand[agent] = True

    def score(self, agent):
        '''Returns the score of an agent's hand.'''
        score, _ = self.score_soft(agent)
        return score

    def busted(self, agent):
        '''Returns True if an agent is busted.'''
        return self.score(agent) == 0

    def soft(self, agent):
        '''Returns True if the agent's score is soft, i.e. made with an ace as 11.'''
        _, soft = self.score_soft(agent)
        return soft

    def score_soft(self, agent):
        '''Computes the score and softness of an agent's hand.

        The score is the value of the hand. A hand is soft if it contains an
        ace being scored as an 11.

        The return value is a pair `(score, soft)` where `score` is the value
        of the hand and `soft` is a boolean which is True for soft scores.
        '''
        aces = 0
        score = 0
        for card in self.hands[agent]:
            if card is 1:
                score += 11
                aces += 1
            else:
                score += min(card, 10)
        while score > 21 and aces > 0:
            score -= 10
            aces -= 1

        if score == 21 and len(self.hands[agent]) == 2:
            # blackjack :)
            return 22, True
        elif score > 21:
            # bust :(
            return 0, False
        else:
            # normal hand
            return score, aces > 0

class Observation:
    '''An observation seen by an agent during a round of Blackjack.

    An observation can be used to sample from the space of possible future
    observations.

    An observation wraps the true state of the simulation. Any agent accessing
    this state is considered to be cheating.
    '''

    def __init__(self, state, agent):
        '''Construct an observation of the given state for some agent.'''
        self._state = deepcopy(state)
        self.agent = agent

        dealer = self._state.dealer
        hidden_card = self._state.hands[dealer][0]
        self._state.shoe.replace(hidden_card)

    def actions(self):
        '''Returns a set of valid actions.'''
        return self._state.actions(self.agent)

    def sample(self, action):
        '''Samples a new observation from this one given some action.'''
        obs = copy(self)
        obs._state = self._state.sample(self.agent, action)
        return obs

    def score(self):
        '''Returns the score of the agent's hand.'''
        return self._state.score(self.agent)

    def busted(self):
        '''Returns True if the agent has busted.'''
        return self._state.busted(self.agent)

    def soft(self):
        '''Returns True if the agent's score is soft, i.e. made with an ace as 11.'''
        return self._state.soft(self.agent)

    def score_soft(self):
        '''Returns `(score, soft)` where
            - `score` is the score of the agents hand, and
            - `soft` is True if the score is soft.
        '''
        return self._state.score_soft(self.agent)

class Agent:
    '''A base class for agents.'''

    @classmethod
    def from_fn(cls, policy):
        '''Constructs an Agent from a policy function.'''
        assert callable(policy), 'policies must be callable'
        agent = cls()
        agent.policy = policy
        return agent

    def policy(self, obs, ctx):
        raise NotImplementedError

    def __call__(self, obs, ctx):
        '''Agents can be called just like plain policy functions.'''
        return self.policy(obs, ctx)

class RandomAgent(Agent):
    '''An agent which behaves randomly.'''
    def policy(self, obs, ctx):
        return random.choice(obs.actions())

class DealerAgent(Agent):
    '''An agent which plays like a casino dealer.'''

    def __init__(self, n=17):
        self.n = n

    def policy(self, obs, ctx):
        score, soft = obs.score_soft()
        if self.n < score:
            return Action.HIT
        elif soft and score == self.n:
            return Action.HIT
        return Action.STAND

class Simulator:
    def __init__(self, *players, dealer=DealerAgent(), n_decks=2, cut=0.5):
        '''Constructs a new Simulator.

        Args:
            *players: The agents for each player.
            dealer: The agent for the dealer.
            n_decks: The number of decks to play with.
            cut: Reshuffle the deck when it's size is below this percent.
        '''
        assert len(players) > 0, 'must have at least one player policy'
        assert 0 <= cut and cut < 1, 'cut must be between 0 and 1'
        self.dealer = dealer if isinstance(dealer, Agent) else Agent.from_fn(dealer)
        self.players = tuple(a if isinstance(a, Agent) else Agent.from_fn(a) for a in players)
        self.shoe = Shoe(n_decks)
        self.n_decks = n_decks
        self.cut = cut

    def run(self, n_rounds):
        '''Execute the simulation for some number of rounds and return a summary.'''
        # The order of play is given by the order of the players, followed by the dealer.
        players = self.players + (self.dealer,)

        # Keep track of win statistics during the simulation.
        wins = np.zeros((n_rounds, len(players)), dtype=bool)
        for i in range(n_rounds):
            logger.info(f'round {i}')

            # Reshuffle if below the cut
            if len(self.shoe) < 13 * self.n_decks * self.cut:
                logger.info('reshuffling')
                self.shoe.reshuffle()

            # Draw the hands and create a state for this round.
            hands = {}
            for agent in players:
                hands[agent] = [self.shoe.draw(), self.shoe.draw()]
            state = State(self.shoe, hands, self.dealer)

            # Let each player play until they stand or bust.
            for agent in players:
                logger.info(f'{agent}\'s turn')
                ctx = {}
                while state.stand[agent] == False:
                    obs = Observation(state, agent)
                    action = agent(obs, ctx)
                    state.next(agent, action)

            # Compute the winners of this round and update the statistics
            winners = None
            best_score = -1
            for agent in players:
                score = state.score(agent)
                if score > best_score:
                    winners = [agent]
                    best_score = score
                elif score == best_score:
                    winners.append(agent)
            for agent in winners:
                j = players.index(agent)
                wins[i,j] = True

        return wins
