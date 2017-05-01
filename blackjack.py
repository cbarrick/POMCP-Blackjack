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
from enum import IntEnum
import logging
import numpy as np
import pandas as pd
import random

logger = logging.getLogger(__name__)


class Action(IntEnum):
    '''The possible actions in a game of Blackjack.

    Enum:
        STAND: Perform no more actions this round.
        HIT: Draw another card.
        DOUBLE: Double your bet, hit, then stand.
    '''
    STAND = 1
    HIT = 2
    # DOUBLE = 3
    # SPLIT = 4

class Shoe:
    '''A shuffled collection of French playing cards.'''

    _INDICIES = np.arange(13)

    def __init__(self, n_decks):
        '''Create a full Shoe with some number of decks of cards.'''
        assert n_decks > 0, '`n_decks` must be greater than 0.'
        self.n_decks = n_decks
        self.counts = np.zeros(13, dtype=int) + 4*n_decks

    def __len__(self):
        '''The length of a shoe is the number of cards.'''
        return np.sum(self.counts)

    def sample(self):
        assert len(self) > 0, 'cannot sample from an empty shoe.'
        i = np.random.choice(Shoe._INDICIES, p=self.counts/np.sum(self.counts))
        new_shoe = deepcopy(self)
        new_shoe.counts[i] -= 1
        card = i + 1
        return card, new_shoe

    def replace(self, card):
        '''Replace a card back into the shoe at a random position.'''
        i = card - 1
        count = self.counts[i]
        max_count = 4 * self.n_decks
        assert count <= max_count, f'cannot have more than {max_count} cards of value {card}.'
        new_shoe = deepcopy(self)
        new_shoe.counts[i] = count + 1
        return new_shoe


class State:
    '''A state of a round of Blackjack.'''

    def __init__(self, shoe, hands, stand):
        '''Create a new state from a given shoe and hands dict.

        Args:
            shoe:
                The shoe to draw from when sampling the next state.
            hands:
                A list of hands for the agents. Hands are represented as nested
                tuples (cons cells). The last hand belongs to the dealer.
            stand:
                A list of boolean states for the agents. True means they stand.
        '''
        self.shoe = shoe
        self.hands = tuple(hands)
        self.stand = tuple(stand or self.busted(j) for j, stand in enumerate(stand))
        self.dealer = len(self.hands) - 1

    @classmethod
    def start_state(cls, shoe, n_agents):
        '''Constructs an initial state starting from the shoe.

        Args:
            shoe: The shoe from which to draw the initial hands.
            n_agents: The number of agents in the game.
        '''
        hands = []
        for i in range(n_agents):
            a, shoe = shoe.sample()
            b, shoe = shoe.sample()
            hand = (a, (b, ()))
            hands.append(hand)

        hands = tuple(hands)
        stand = tuple(False for i in range(n_agents))
        return cls(shoe, hands, stand)

    def hidden_card(self):
        '''Returns the dealer's hidden card.'''
        hand = self.hands[self.dealer]
        while hand is not ():
            card, hand = hand
        return card

    def sample(self, agent, action):
        '''Sample a new state from this state by commiting an action as a player.'''

        assert isinstance(action, Action), 'action must be an instance of blackjack.Action'
        assert not self.stand[agent], 'cannot perform action after stand or bust'

        card, shoe = self.shoe.sample()

        if action is Action.HIT:
            new_hand = (card, self.hands[agent])
        else:
            new_hand = self.hands[agent]

        if action is Action.STAND:
            new_stand = True
        else:
            new_stand = self.stand[agent]

        hands = (new_hand if i is agent else hand for i, hand in enumerate(self.hands))
        stand = (new_stand if i is agent else stand for i, stand in enumerate(self.stand))
        return State(shoe, hands, stand)

    def get_observation(self, agent, is_dealer):
        '''Returns the observation for the given agent.'''
        return Observation(self, agent, is_dealer)

    def actions(self, agent):
        '''Returns the set of valid actions for the given agent.'''
        return tuple(Action) if not self.stand[agent] else ()

    def score(self, agent):
        '''Returns the score of an agent's hand.'''
        score, _, _ = self.score_soft_busted(agent)
        return score

    def busted(self, agent):
        '''Returns True if an agent is busted.'''
        _, _, busted = self.score_soft_busted(agent)
        return busted

    def soft(self, agent):
        '''Returns True if the agent's score is soft, i.e. made with an ace as 11.'''
        _, soft, _ = self.score_soft_busted(agent)
        return soft

    def score_soft_busted(self, agent):
        '''Computes the score and softness of an agent's hand.

        The score is the value of the hand. A hand is soft if it contains an
        ace being scored as an 11.

        Returns `(score, soft, busted)` where
            - `score` is the score of the agents hand, and
            - `soft` is True if the score is soft.
            - `busted` is True if the agent has busted
        '''
        aces = 0
        score = 0

        hand = self.hands[agent]
        while hand is not ():
            card, hand = hand
            if card is 1:
                aces += 1
                score += 11
            elif card > 10:
                score += 10
            else:
                score += card

        while score > 21 and aces > 0:
            score -= 10

        soft = (aces > 0)
        if soft and score is 21:
            score = 22
        elif score > 21:
            score = 0
        busted = score == 0

        return score, soft, busted


class Observation:
    '''An observation seen by an agent during a round of Blackjack.

    An observation can be used to sample from the space of possible future
    observations.

    An observation wraps the true state of the simulation. Any agent accessing
    this state is considered to be cheating.
    '''

    def __init__(self, state, agent, is_dealer):
        '''Construct an observation of the given state for some agent.'''
        hidden_card = state.hidden_card()
        if not is_dealer:
            state = copy(state)
            state.shoe = state.shoe.replace(hidden_card)

        self._state = state
        self.agent = agent

    def sample_belief(self):
        '''Sample a belief state from this observation.'''
        return Belief.from_observation(self)

    def actions(self):
        '''Returns a set of valid actions.'''
        return self._state.actions(self.agent)

    def score(self):
        '''Returns the score of the agent's hand.'''
        return self._state.score(self.agent)

    def busted(self):
        '''Returns True if the agent has busted.'''
        return self._state.busted(self.agent)

    def soft(self):
        '''Returns True if the agent's score is soft, i.e. made with an ace as 11.'''
        return self._state.soft(self.agent)

    def score_soft_busted(self):
        '''Returns `(score, soft, busted)` where
            - `score` is the score of the agents hand, and
            - `soft` is True if the score is soft.
            - `busted` is True if the agent has busted
        '''
        return self._state.score_soft_busted(self.agent)


class Belief:
    def __init__(self, state, hidden_card, agent):
        state = copy(state)
        hidden_card, state.shoe = state.shoe.sample()

        self._state = state
        self.hidden_card = hidden_card
        self.agent = agent

    @classmethod
    def from_observation(cls, obs):
        '''Construct a belief state from an observation.'''
        state = copy(obs._state)
        hidden_card, state.shoe = state.shoe.sample()
        return cls(state, hidden_card, obs.agent)

    def sample(self, action):
        '''Sample a possible future belief state.'''
        next_state = self._state.sample(self.agent, action)
        return Belief(next_state, self.hidden_card, self.agent)

    def actions(self):
        '''Returns a set of valid actions.'''
        return self._state.actions(self.agent)

    def score(self):
        '''Returns the score of the agent's hand.'''
        return self._state.score(self.agent)

    def busted(self):
        '''Returns True if the agent has busted.'''
        return self._state.busted(self.agent)

    def soft(self):
        '''Returns True if the agent's score is soft, i.e. made with an ace as 11.'''
        return self._state.soft(self.agent)

    def score_soft_busted(self):
        '''Returns `(score, soft)` where
            - `score` is the score of the agents hand, and
            - `soft` is True if the score is soft.
        '''
        return self._state.score_soft_busted(self.agent)


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


class RandomAgent(Agent):
    '''An agent which behaves randomly.'''
    def policy(self, obs, ctx):
        return random.choice(obs.actions())


class DealerAgent(Agent):
    '''An agent which plays like a casino dealer.'''

    def __init__(self, n=17):
        self.n = n

    def __str__(self):
        return f"Dealer {self.n}"

    def policy(self, obs, ctx):
        score, soft, busted = obs.score_soft_busted()
        if score < self.n:
            return Action.HIT
        elif busted:
            return Action.STAND
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
        self.start_shoe = Shoe(n_decks)
        self.n_decks = n_decks
        self.cut = cut

    def run(self, n_rounds):
        '''Execute the simulation for some number of rounds and return a summary.'''
        agents = self.players + (self.dealer,)
        scores = np.zeros((n_rounds, len(agents)), dtype=int)
        state = State.start_state(self.start_shoe, len(agents))

        for i in range(n_rounds):
            # Start state:
            # Reshuffle if below the cut, i.e. create a start state from the start shoe.
            # Otherwise create the start state reusing the previous shoe.
            if len(state.shoe) < 13 * self.n_decks * self.cut:
                state = State.start_state(self.start_shoe, len(agents))
            else:
                state = State.start_state(state.shoe, len(agents))

            # Let each player play until they stand or bust.
            for j, agent in enumerate(agents):
                ctx = {}
                while state.stand[j] == False:
                    obs = state.get_observation(j, agent is self.dealer)
                    action = agent.policy(obs, ctx)
                    state = state.sample(j, action)
                scores[i,j] = state.score(j)

        return pd.DataFrame(scores, columns=[agent for agent in agents])
