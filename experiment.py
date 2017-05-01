import pomcp
import blackjack

agent = pomcp.POMCP()
game = blackjack.Simulator(agent)
stats = game.run(10)
print(stats)
