import numpy as np

class NeuralNet(object):
	def __init__(self,input_size,hidden_size,output_size):
		self.W1 = np.random.randn(input_size,hidden_size) * np.sqrt(2.0/input_size)
		self.W2 = np.random.randn(hidden_size,output_size) * np.sqrt(2.0/hidden_size)
		self.b1 = np.zeros(hidden_size)
		self.b2 = np.zeros(output_size)

	def loss(self,X,y,reg):

		num_train, num_features = X.shape
		num_classes = np.max(y) + 1

		W1 = self.W1
		W2 = self.W2
		b1 = self.b1
		b2 = self.b2

		p = 0.6
		hidden = np.maximum(0,X.dot(W1) + b1) 
		U =	(np.random.rand(*hidden.shape) < p) / p 
		hidden *= U 
		scores = hidden.dot(W2) + b2

		# LOSS
		loss = 0.0
		exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
		probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
		correct_log_scores = -np.log(probs[range(num_train),y])
		loss = np.sum(correct_log_scores)
		loss /= num_train
		# loss += 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)

		# GRADIENT
		grads = {}
		dloss = 1.0

		dscores = dloss * probs
		dscores[range(num_train),y] -= 1
		dscores /= num_train

		db2 = 1.0 * np.sum(dscores,axis=0)

		dW2 = hidden.T.dot(dscores)
		dhidden = dscores.dot(W2.T)
		dhidden[hidden <= 0] = 0

		db1 = 1.0 * np.sum(dhidden,axis=0)
		dW1 = X.T.dot(dhidden)

		# dW1 += reg*W1
		# dW2 += reg*W2

		grads['W1'] = dW1
		grads['W2'] = dW2
		grads['b1'] = db1
		grads['b2'] = db2

		return loss,grads

	def train(self,X_train,y_train,X_val,y_val,reg,learning_rate,learning_rate_decay=0.95,batch_size=200,num_iters=200,verbose=True):

		num_train, num_features = X_train.shape
		num_classes = np.max(y_train) + 1

		iterations_per_epoch = np.maximum(1,num_train/batch_size)

		loss_history = []
		train_acc_history = []
		val_acc_history = []

		m1, m2, m3, m4 = 0,0,0,0
		v1, v2, v3, v4 = 0,0,0,0
		beta1 = 0.9
		beta2 = 0.99
		eps = 1e-7

		for it in range(num_iters):
			indices = np.random.choice(num_train,batch_size,replace=True)
			X_batch = X_train[indices]
			y_batch = y_train[indices]

			loss, grads = self.loss(X_batch,y_batch,reg)
			loss_history.append(loss)
			
			# UPDATE
			# self.W1 -= learning_rate*grads['W1']
			# self.W2 -= learning_rate*grads['W2']
			# self.b1 -= learning_rate*grads['b1']
			# self.b2 -= learning_rate*grads['b2']

			m1 = beta1*m1 + (1-beta1)*grads['W1']
			v1 = beta2*v1 + (1-beta2)*(grads['W1']**2)
			self.W1 += (-learning_rate*m1) / (np.sqrt(v1) + eps)

			m2 = beta1*m2 + (1-beta1)*grads['W2']
			v2 = beta2*v2 + (1-beta2)*(grads['W2']**2)
			self.W2 += (-learning_rate*m2) / (np.sqrt(v2) + eps)

			m3 = beta1*m3 + (1-beta1)*grads['b1']
			v3 = beta2*v3 + (1-beta2)*(grads['b1']**2)
			self.b1 += (-learning_rate*m3) / (np.sqrt(v3) + eps)

			m4 = beta1*m4 + (1-beta1)*grads['b2']
			v4 = beta2*v4 + (1-beta2)*(grads['b2']**2)
			self.b2 += (-learning_rate*m4) / (np.sqrt(v4) + eps)

			if verbose and it%100 == 0:
				print "Loss in Iteration (%d/%d): %f" % (it,num_iters,loss)

			if it%iterations_per_epoch == 0:
				learning_rate *= learning_rate_decay
				train_acc = np.mean(self.predict(X_batch) == y_batch)
				val_acc = np.mean(self.predict(X_val) == y_val)
				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

		return {
			'loss_history':loss_history,
			'train_acc_history':train_acc_history,
			'val_acc_history':val_acc_history
		}

	def predict(self,X):
		hidden = np.maximum(0,X.dot(self.W1) + self.b1)
		scores = hidden.dot(self.W2) + self.b2
		y = np.argmax(scores,axis=1)
		return y



