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

N = [10,100,1000,10000]
n_decks = [6,7,8,9,10]
columns = ['N','Decks','POMCP','Random','Agent16','Agent17','Agent18']
trial_num=[]
deck_num=[]
pomcp_win =[]
rand_win =[]
a16_win =[]
a17_win =[]
a18_win =[]



for trials in N:
    for deck in n_decks:
        game = blackjack.Simulator(*agents,dealer=dealer,n_decks=deck)
        outcomes = game.run(trials)

        trial_num.append(trials)
        deck_num.append(deck)
        dealer_scores = outcomes[dealer]
        pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/trials)
        rand_win.append(sum(outcomes[a1] > dealer_scores)/trials)
        a16_win.append(sum(outcomes[a2] > dealer_scores)/trials)
        a17_win.append(sum(outcomes[a3] > dealer_scores)/trials)
        a18_win.append(sum(outcomes[a4] > dealer_scores)/trials)

scores = pd.DataFrame({'N':trial_num,
                       'Decks': deck_num,
                       'POMCP':pomcp_win,
                       'Random':rand_win,
                       'Agent16':a16_win,
                       'Agent17':a17_win,
                       'Agent18':a18_win})

print(scores)
