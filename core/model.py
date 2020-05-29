import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.models import Sequential, load_model
import tensorflow as tf 

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = tf.keras.models.load_model(filepath)

	def predict_generator(self, data_gen, steps=0):
    # Predict out of memory
		predicted = self.model.predict_generator(
			data_gen,
			steps=steps,
			verbose=1
		)
		return predicted


	def get_next_n_predictions(self, data, n, verbose=False):
		'''
		Get a window of the next n predicted data
		'''
		predictions = []
		
		for p in range(n):
			predicted_data = self.model.predict(data, verbose=verbose)

			tmp_predictions = []

			for i in range(predicted_data.shape[0]):
				pred = predicted_data[i].reshape(1, -1)
				data[i] = np.append(data[i], pred, axis=0)[1:]
				tmp_predictions.append(pred[0])

			predictions.append(tmp_predictions)
		
		return np.array(predictions)


	def predict_point_by_point(self, data, verbose=0):
		'''
    Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		'''
		predicted = self.model.predict(data, verbose=verbose)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted

	def predict_sequences_multiple(self, data, window_size, prediction_len):
		'''
    Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    '''
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs

	def predict_sequence_full(self, data, window_size):
		'''
    Shift the window by 1 new prediction each time, re-run predictions on new window
    '''
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
