# %%

%run rsa/message_rsa.py

# %%

from exh import *

# %%
################## M IMPLICATURES  #############################

frequent   = a
infrequent = a & b

universe = Universe(fs = [a]) 

rsa = MessageRSA(universe, 
                 messages = [
                             ("*nothing*",       [a | ~a]),
                             ("short",   [a, ~a]),
                             ("long",    [a, ~a])
                            ],
                 states = [
                     ("frequent",    a),
                     # ("?",    a | ~a),
                     ("infrequent", ~a)
                  ],
                 costs = np.array([5, 1, 1.1], dtype = "float"),
                 world_prior = np.array([0.3, 0.7]),
                 rationality = 1.)



# %%

rsa.compute(6)

# %%

for i in range(len(rsa.listeners)):
    rsa.heatmap_listener(i)

# %%

rsa.heatmap_speaker()

# %%

#P(s | u) = P(u | s) * P(s)
state_given_utterance = rsa.last_speaker *  rsa.state_prior[:,np.newaxis]
heatmap(state_given_utterance, rsa.lab_states, rsa.lab_messages)
state_given_utterance = (state_given_utterance / np.sum(state_given_utterance, axis = 0)).transpose()

heatmap(state_given_utterance, rsa.lab_messages, rsa.lab_states)
# %%
# P(w|u) = sum P(w | u, s)P(s | u) = sum P(w|s) * P(s|u)
listener = np.sum(rsa.states[np.newaxis, ...] *  state_given_utterance[..., np.newaxis], axis = 1)

"""
nothing x y 
a       z z"
~a      

"""

# %%
rsa.heatmap_listener()

# %%


# %%
speaker = rsa.last_speaker

# P(message | state)
# TODO: lexicon prior
speaker_given_state_only =  np.sum(speaker, axis = 1)  

# P(state | message)
state_given_message = speaker_given_state_only.transpose() *  np.sum(rsa.states.dot(np.diag(rsa.prior)), axis = 1)[np.newaxis, :]
state_given_message = state_given_message / np.sum(state_given_message, axis = -1)[:, np.newaxis]

heatmap(state_given_message, rsa.lab_messages, rsa.lab_states)

# prior_world_state = prior_world_state / np.sum(prior_world_state, axis = -1)[:, np.newaxis]


listener = state_given_message.dot(rsa.states)


heatmap(listener, rsa.lab_messages, rsa.lab_worlds)


# %%

heatmap(prior_world_state, rsa.lab_states, rsa.lab_worlds)

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


