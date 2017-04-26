import pomcp
import blackjack

agent = pomcp.POMCP()
game = blackjack.Simulator(agent)
print(game.run(10))
