# %%
from exh import *
from exh.worlds import Universe
import numpy as np
import scipy.stats as stats

# %%

def table(np_array, lab_lines, lab_cols, round_to = 3):
	table = Table(char_bold_col = "||")
	table.set_header([""] + list(lab_cols))
	table.set_strong_col(1)
	rounded_array = np.round(np_array, round_to)

	for line, label in zip(rounded_array, lab_lines):
		table.add_row([label] + [str(value) for value in line]) 

	table.print_console()


# %%

d = Pred(4, name = "d")

# %%

prop_universe = Universe(fs = [a, b, c, d])
prop_universe.evaluate(a & b)


# %%
############# Listener 0

rationality = 1.

universe  = Universe(fs = [a, b])
alts      = [a | b, a, b, a & b, ~a]

n_alts   = len(alts)
n_worlds = universe.n_worlds
n_states = n_worlds

# P(w|s)
states = np.identity(universe.n_worlds)
costs = np.ones(n_alts)


lab_worlds = ["".join([str(i) for i, val in enumerate(world) if val]) for world in universe.worlds]
# lab_states = lab_worlds[:]
lab_states = ["", "a", "b", "ab"]
lab_alts = list(map(lambda x: x.display(False), alts)) 

# world priors
# prior = np.ones(universe.worlds.shape[0])
prior = np.arange(universe.worlds.shape[0]) + 1
prior = prior / np.sum(prior)

# literal listener
# P(w | u)
# universe.worlds[i_utterance, i_world] 

listener0 = universe.evaluate(*alts).transpose() * prior
listener0 = listener0 / np.sum(listener0, axis = 1)[:, np.newaxis]

print("listener0 listener")
table(listener0, lab_alts, lab_worlds)


#%%
############# Speaker 0
listener_with_states   = np.repeat(listener0[np.newaxis, ...], n_states, axis = 0)
states_with_utterances = np.repeat(states[:, np.newaxis, :], n_alts, axis = 1)

print(listener_with_states.shape)
print(states_with_utterances.shape)

utility = - (stats.entropy(states_with_utterances, 
                         listener_with_states, axis=2) + costs[np.newaxis,:])

# P(u | s)
speaker1 = np.exp(rationality * utility)
speaker1 = speaker1 / np.sum(speaker1, axis = 1)[:,np.newaxis]

table(speaker1, lab_states, lab_alts)


# %%
############### Listener 1

# P(s|u) prop P(u|s)
state_given_utterance = (speaker1 / np.sum(speaker1, axis = 0)).transpose()
table(state_given_utterance, lab_alts, lab_states)

# P(w|u) = sum P(w | u, s)P(s | u) = sum P(w|s) * P(s|u)
listener1 = np.sum(states[np.newaxis, ...] *  state_given_utterance[..., np.newaxis], axis = 1)
table(listener1, lab_alts, lab_worlds)

# %%
############### Speaker 2

listener_with_states   = np.repeat(listener1[np.newaxis, ...], n_states, axis = 0)
states_with_utterances = np.repeat(states[:, np.newaxis, :], n_alts, axis = 1)

print(listener_with_states.shape)
print(states_with_utterances.shape)

utility = - (stats.entropy(states_with_utterances, 
                         listener_with_states, axis=2) + costs[np.newaxis, :])

# P(u | s)
speaker2 = np.exp(rationality * utility)
speaker2 = speaker2 / np.sum(speaker2, axis = 1)[:, np.newaxis]

table(speaker2, lab_states, lab_alts)

# %%
############### Listener 2

# P(s|u) prop P(u|s)
state_given_utterance = (speaker2 / np.sum(speaker2, axis = 0)).transpose()
# table(state_given_utterance, lab_alts, lab_states)

# P(w|u) = sum P(w | u, s)P(s | u) = sum P(w|s) * P(s|u)
listener2 = np.sum(states[np.newaxis, ...] *  state_given_utterance[..., np.newaxis], axis = 1)
table(listener2, lab_alts, lab_worlds)

# %%
############### Speaker 3

listener_with_states   = np.repeat(listener2[np.newaxis, ...], n_states, axis = 0)
states_with_utterances = np.repeat(states[:, np.newaxis, :], n_alts, axis = 1)

print(listener_with_states.shape)
print(states_with_utterances.shape)

utility = - (stats.entropy(states_with_utterances, 
                         listener_with_states, axis=2) + costs[np.newaxis, :])

# P(u | s)
speaker3 = np.exp(rationality * utility)
speaker3 = speaker3 / np.sum(speaker3, axis = 1)[:, np.newaxis]

table(speaker3, lab_states, lab_alts)





# speaker_with_worlds = np.repeat(speaker1[np.newaxis, ], n_worlds, axis = )



# %%
