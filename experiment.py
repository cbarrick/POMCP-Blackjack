import pomcp
import blackjack

agent_pomcp = pomcp.POMCP()
game_p = blackjack.Simulator(agent, n_decks=3)
print(game_p.run(10))
#get wins
#get draws
#get losses
