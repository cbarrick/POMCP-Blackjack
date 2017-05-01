import pomcp
import blackjack

N = 1000

pomcp_agent = pomcp.POMCP()
a1 = blackjack.RandomAgent()
a2 = blackjack.DealerAgent(n=16)
a3 = blackjack.DealerAgent(n=17)
a4 = blackjack.DealerAgent(n=18)
agents = [a1, a2, pomcp_agent, a3, a4]
dealer = blackjack.DealerAgent()
game = blackjack.Simulator(*agents, dealer=dealer, n_decks=4)
outcomes = game.run(N)

pomcp_scores = outcomes[pomcp_agent]
dealer_scores = outcomes[dealer]

wins = pomcp_scores > dealer_scores
losses = (pomcp_scores < dealer_scores) | pomcp_scores.apply(lambda score: score == 0)
draws = (pomcp_scores == dealer_scores) & pomcp_scores.apply(lambda score: score > 0)

n_wins = sum(wins)
n_losses = sum(losses)
n_draws = sum(draws)

pomcp_scores.apply(lambda score: score > 0)

print('# Stats:')
print(f'Win %:   {n_wins / N:0.1%}')
print(f'Loss %:  {n_losses / N:0.1%}')
print(f'Draw %:  {n_draws / N:0.1%}')
print()

print('# Raw Outcomes:')
print(outcomes)
