import pomcp
import blackjack
import pandas as pd


pomcp_agent = pomcp.POMCP()
a1 = blackjack.RandomAgent()
a2 = blackjack.DealerAgent(n=16)
a3 = blackjack.DealerAgent(n=17)
a4 = blackjack.DealerAgent(n=18)
dealer = blackjack.DealerAgent()
agents = [a1, a2, pomcp_agent, a3, a4]
pomcp_a1 = [a1,pomcp_agent]
pomcp_a2 = [a2,pomcp_agent]
pomcp_a3 = [a3,pomcp_agent]
pomcp_a4 = [a4,pomcp_agent]


N = 5000
n_decks = [5,6,7,8,9,10,11,12,13]



''' Get average wins with random agent, pomcp, and dealer'''
pomcp_win =[]
rand_win =[]
for deck in n_decks:
    game = blackjack.Simulator(*pomcp_a1,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    rand_win.append(sum(outcomes[a1] > dealer_scores)/N)


scores1 = pd.DataFrame(columns=['Decks','POMCP','Random'])
scores1['Decks']   = n_decks
scores1['POMCP']   = pomcp_win
scores1['Random']  = rand_win
print('POMCP v RandomAgent')
print(scores1)
print('\n')



''' Get average wins with dealer agent 16, pomcp, and dealer'''
pomcp_win =[]
a16_win =[]
for deck in n_decks:
    game = blackjack.Simulator(*pomcp_a2,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    a16_win.append(sum(outcomes[a2] > dealer_scores)/N)


scores2 = pd.DataFrame(columns=['Decks','POMCP','Agent16'])
scores2['Decks']   = n_decks
scores2['POMCP']   = pomcp_win
scores2['Agent16']  = a16_win
print('POMCP v DealerAgent(n=16)')
print(scores2)
print('\n')



''' Get average wins with dealer agent 17, pomcp, and dealer'''
pomcp_win =[]
a17_win =[]
for deck in n_decks:
    game = blackjack.Simulator(*pomcp_a3,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    a17_win.append(sum(outcomes[a3] > dealer_scores)/N)


scores3 = pd.DataFrame(columns=['Decks','POMCP','Agent17'])
scores3['Decks']   = n_decks
scores3['POMCP']   = pomcp_win
scores3['Agent17']  = a17_win
print('POMCP v DealerAgent(n=17)')
print(scores3)
print('\n')

''' Get average wins with dealer agent 18, pomcp, and dealer'''
pomcp_win =[]
a18_win =[]
for deck in n_decks:
    game = blackjack.Simulator(*pomcp_a4,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    a18_win.append(sum(outcomes[a4] > dealer_scores)/N)


scores4 = pd.DataFrame(columns=['Decks','POMCP','Agent18'])
scores4['Decks']   = n_decks
scores4['POMCP']   = pomcp_win
scores4['Agent18']  = a17_win
print('POMCP v DealerAgent(n=18)')
print(scores4)
print('\n')



''' Get average wins with 4 agents, pomcp, and dealer'''
pomcp_win =[]
rand_win =[]
a16_win =[]
a17_win =[]
a18_win =[]
for deck in n_decks:
    game = blackjack.Simulator(*agents,dealer=dealer,n_decks=deck)
    outcomes = game.run(N)
    dealer_scores = outcomes[dealer]
    pomcp_win.append(sum(outcomes[pomcp_agent] > dealer_scores)/N)
    rand_win.append(sum(outcomes[a1] > dealer_scores)/N)
    a16_win.append(sum(outcomes[a2] > dealer_scores)/N)
    a17_win.append(sum(outcomes[a3] > dealer_scores)/N)
    a18_win.append(sum(outcomes[a4] > dealer_scores)/N)

scores = pd.DataFrame(columns=['Decks','POMCP','Random','Agent16','Agent17','Agent18'])
scores['Decks']   = n_decks
scores['POMCP']   = pomcp_win
scores['Random']  = rand_win
scores['Agent16'] = a16_win
scores['Agent17'] = a17_win
scores['Agent18'] = a18_win
print('POMCP v All Agents')
print(scores)
