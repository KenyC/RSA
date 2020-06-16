import numpy             as np
import matplotlib.pyplot as plt
import scipy.stats       as stats

from exh.utils.table import Table
from utils import *

class MessageRSA:

	def __init__(self, universe, messages, states = None, world_prior = None, costs = None, rationality = 1.):
		self.universe = universe
		self.rationality = rationality
		self.messages = messages

		self.n_messages   = len(messages)
		self.n_worlds = self.universe.n_worlds

		if costs is None:
			costs = np.ones(self.n_messages)
		self.costs = costs

		# Labels
		name_worlds = self.universe.name_worlds()
		self.lab_worlds = ["".join([name for name, val in zip(name_worlds, world) if val]) for world in universe.worlds]
		# lab_states = lab_worlds[:]
		if states is None:
			self.lab_states = self.lab_worlds[:]
		else:
			self.lab_states = list(map(lambda x: x[0], states)) 
		self.lab_messages = list(map(lambda x: x[0], self.messages)) 

		# message truth-conditions matrix
		self.lab_tcs            = [tc.display(False) for tc in self.truth_conditions()]
		self.n_tcs              = sum(len(val) for _, val in self.messages)

		self.message_tcs_matrix = np.zeros((self.n_messages, self.n_tcs), dtype = "float")
		offset = 0
		for i, (_, val) in enumerate(self.messages):
			self.message_tcs_matrix[i, offset:offset + len(val)] = 1
			offset += len(val)

		self.costs_tcs          = self.message_tcs_matrix.transpose().dot(self.costs)
		self.message_tcs_matrix = self.message_tcs_matrix / np.sum(self.message_tcs_matrix, axis = 1) [:, np.newaxis]



		# world priors
		# prior = np.ones(universe.worlds.shape[0])
		if world_prior is None:
			world_prior = np.ones(universe.worlds.shape[0]) + 1
		self.prior = world_prior / np.sum(world_prior)


		if states is None:
			self.states = np.identity(self.n_worlds)
		else:
			self.states = np.stack(self.universe.evaluate(*(map(lambda x: x[1], states)))).transpose()

		self.states   = self.states / np.sum(self.states, axis = 1) [:, np.newaxis]
		self.n_states = len(self.states)

		self.compute(n = 0)

	

	def compute(self, n = 1, reset = True):
		if reset:
			self.speakers = []
			self.listeners = []

		self.compute_listener = len(self.listeners) == len(self.speakers)

		for _ in range(n):
			if self.compute_listener:
				if self.listeners:
					if len(self.listeners) > 1:
						self.compute_rational_listener()
					else:
						self.compute_first_rational_listener()
				else:
					self.compute_literal_listener()
			else:
				if self.speakers:
					self.compute_rational_speaker()
				else:
					self.compute_literal_speaker()

			self.compute_listener = not self.compute_listener




	def compute_literal_listener(self):
		listener0 = self.universe.evaluate(*self.truth_conditions()).transpose() * self.prior
		listener0 = listener0 / np.sum(listener0, axis = 1)[:, np.newaxis]

		self.listeners.append(listener0)


	def compute_literal_speaker(self):
		# shape : (states, tcs, worlds)
		listener_with_states   = np.repeat(self.last_listener[np.newaxis, ...], self.n_states, axis = 0)
		states_with_utterances = np.repeat(self.states[:, np.newaxis, :],       self.n_tcs,   axis = 1)

		utility = - (stats.entropy(states_with_utterances, 
		                           listener_with_states, axis=2) + self.costs_tcs)

		# P(u | s)
		speaker = np.exp(self.rationality * utility)
		speaker = speaker / np.sum(speaker, axis = 1)[:, np.newaxis]

		self.speakers.append(speaker)


	def compute_first_rational_listener(self):
		# P(u | v) = P(v | u) * P(u) / Sum P(v | u) P(u)
		state_given_message_and_lexicon = (self.last_speaker / np.sum(self.last_speaker, axis = 0)).transpose()


		# P(message | state) = sum_lexicon P(message, lexicon | state) P(lexicon | state)  
		# message_tcs_matrix : (messages, tcs)
		# state_given_message_and_lexicon : (tcs, state)
		state_given_message = np.sum(self.message_tcs_matrix[..., np.newaxis] *  state_given_message_and_lexicon[np.newaxis, ...], axis = 1)
		listener = np.sum(self.states[np.newaxis, ...] *  state_given_message[..., np.newaxis], axis = 1)

		self.listeners.append(listener)



	def compute_rational_speaker(self):
		listener_with_states   = np.repeat(self.last_listener[np.newaxis, ...], self.n_states, axis = 0)
		states_with_utterances = np.repeat(self.states[:, np.newaxis, :],       self.n_messages,   axis = 1)
		print(listener_with_states.shape, states_with_utterances.shape)

		utility = - (stats.entropy(states_with_utterances, 
		                         listener_with_states, axis=2) + self.costs[np.newaxis,:])
		# table(utility, lab_cols = lab_messages, lab_lines = lab_messages)

		# P(u | s)
		speaker = np.exp(self.rationality * utility)
		speaker = speaker / np.sum(speaker, axis = 1)[:,np.newaxis]

		self.speakers.append(speaker)




	def compute_rational_listener(self):
		state_given_utterance = (self.last_speaker / np.sum(self.last_speaker, axis = 0)).transpose()

		# P(w|u) = sum P(w | u, s)P(s | u) = sum P(w|s) * P(s|u)
		listener = np.sum(self.states[np.newaxis, ...] *  state_given_utterance[..., np.newaxis], axis = 1)

		self.listeners.append(listener)


	def display(self, n = -1, listener = True, fn = table, round_to = 3):
		if listener:
			norm_n = n % len(self.listeners)
			list_dist = self.listeners

			if norm_n == 0:
				lab_lines = self.lab_tcs
				lab_cols  = self.lab_worlds
			else:
				lab_lines = self.lab_messages
				lab_cols  = self.lab_worlds

		else:
			norm_n = n % len(self.speakers)
			list_dist = self.speakers

			if norm_n == 0:
				lab_cols = self.lab_tcs
				lab_lines  = self.lab_states
			else:
				lab_cols = self.lab_messages
				lab_lines  = self.lab_states


		fn(list_dist[n], lab_lines, lab_cols, round_to = round_to)


	def table_listener(self, n = -1, **kwargs):
		self.display(n, True, table, **kwargs)

	def table_speaker(self, n = -1, **kwargs):
		self.display(n, False, table, **kwargs)

	def heatmap_listener(self, n = -1, **kwargs):
		self.display(n, True,  heatmap, **kwargs)

	def heatmap_speaker(self, n = -1, **kwargs):
		self.display(n, False, heatmap, **kwargs)



	def truth_conditions(self):
		for _, tcs in self.messages:
			for tc in tcs:
				yield tc



	@property
	def last_listener(self):
		return self.listeners[-1]
	
	@property
	def last_speaker(self):
		return self.speakers[-1]
