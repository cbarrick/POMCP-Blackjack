import pomcp
import blackjack
import pandas as pd


pomcp_agent = pomcp.POMCP()
a1 = blackjack.RandomAgent()
a2 = blackjack.DealerAgent(n=16)
a3 = blackjack.DealerAgent(n=17)
a4 = blackjack.DealerAgent(n=18)
agents = [a1, a2, pomcp_agent, a3, a4]
dealer = blackjack.DealerAgent()

N = 10000
n_decks = [5,6,7,8,9,10]
columns = ['N','Decks','POMCP','Random','Agent16','Agent17','Agent18']
trial_num=[]
deck_num=[]
pomcp_win =[]
rand_win =[]
a16_win =[]
a17_win =[]
a18_win =[]




for deck in n_decks:
    game = blackjack.Simulator(*agents,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)

    deck_num.append(deck)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    rand_win.append(sum(outcomes[a1] > dealer_scores)/N)
    a16_win.append(sum(outcomes[a2] > dealer_scores)/N)
    a17_win.append(sum(outcomes[a3] > dealer_scores)/N)
    a18_win.append(sum(outcomes[a4] > dealer_scores)/N)

scores = pd.DataFrame({'Decks': deck_num,
                       'POMCP':pomcp_win,
                       'Random':rand_win,
                       'Agent16':a16_win,
                       'Agent17':a17_win,
                       'Agent18':a18_win})

print(scores)
