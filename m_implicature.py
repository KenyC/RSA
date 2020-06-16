# %%

%run rsa/message_rsa.py

# %%

from exh import *

# %%
################## M IMPLICATURES  #############################

frequent   = a
infrequent = a & b

universe = Universe(fs = [a, b]) 

rsa = MessageRSA(universe, 
                 messages = [
                             ("",       [a | ~a]),
                             ("a",   [a, a & b, a & ~b]),
                             ("b or (a and b)",    [a & b, b])
                            ],
                 states = [
                     ("a and maybe b",  a),
                     ("a and b",           a & b)
                  ],
                 costs = np.array([5, 1, 1], dtype = "float"),
                 rationality = 1.)



# %%

rsa.compute(2)

# %%

rsa.heatmap_speaker()

# %%

listener = rsa.last_listener
rsa.heatmap_listener()
# %%
listener_given_lexica = rsa.lexicon_matrix.dot(listener)
# listener_given_lexica = listener_given_lexica.dot(rsa.states.transpose())


print(listener_given_lexica)
print(listener_given_lexica.shape)
print(rsa.n_states)

# %%

listener_given_lexica_states   = np.tile(listener_given_lexica,             (rsa.n_states, 1, 1, 1))
states_with_utterances         = np.tile(rsa.states[:, np.newaxis, np.newaxis, :], (1, rsa.n_lexica, rsa.n_messages, 1))

print(listener_given_lexica_states.shape)
print(states_with_utterances.shape)

# %%

utility = - (stats.entropy(states_with_utterances, 
                           listener_given_lexica_states, axis = -1) + rsa.costs[np.newaxis, np.newaxis, :])

# utility = - (stats.entropy(states_with_utterances, 
#                            listener_with_states, axis=2) + self.costs_tcs)

# P(u | s)
speaker = np.exp(rsa.rationality * utility)
speaker = speaker / np.sum(speaker, axis = -1)[..., np.newaxis]

# print(np.sum(speaker, axis = -1))

# %%
idx_lexicon = 1
print(np.exp(utility[:,idx_lexicon,:]) / np.sum(np.exp(utility[:,idx_lexicon,:])))
print(listener_given_lexica[idx_lexicon])


