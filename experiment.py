import pomcp
import blackjack

N = 100

agent = pomcp.POMCP()
dealer = blackjack.DealerAgent()
game = blackjack.Simulator(agent, dealer=dealer)
outcomes = game.run(N)

pomcp_scores = outcomes[agent]
dealer_scores = outcomes[dealer]

wins = pomcp_scores > dealer_scores
losses = (pomcp_scores < dealer_scores) | pomcp_scores.apply(lambda score: score == 0)
draws = (pomcp_scores == dealer_scores) & pomcp_scores.apply(lambda score: score > 0)

n_wins = sum(wins)
n_losses = sum(losses)
n_draws = sum(draws)

pomcp_scores.apply(lambda score: score > 0)

print('# Stats:')
print(f'Win %:   {n_wins / N:0.2%}')
print(f'Loss %:  {n_losses / N:0.2%}')
print(f'Draw %:  {n_draws / N:0.2%}')
print()

print('# Raw Outcomes:')
print(outcomes)
