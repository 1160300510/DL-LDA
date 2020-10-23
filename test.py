import preprocess.pre as pre
import re
import pandas as pd
df = pd.read_csv('data/nlp.csv')
def isRL(str):
    a = re.search(r'^(reinforcement-learning|reinforcement learning|montecarlo|monte carlo|q-learning|q learning'
                  r'|sarsa|dqn|ddpg|a3c|trpo|ppo|td|sac|transfer-learning|transfer learning)$',str)
    if a is None:
        return False
    else:
        return True
def isRNN(str):
    a = re.search(r'\b(rnn|recurrent-neural-network|recurrent neural network|lstm|fully recurrent|elman network'
                  r'|jordan network|hopfield|echo state|recursive neural network|gru|esn|neural turing machine'
                  r'|memory network)\b',str)
    if a is None:
        return False
    else:
        return True
str = pre.processbody(str(df.at[29403,'Body']))
# str = "pgruwaaw"
# str = "we gru iis"
print(str)
print(isRNN(str))