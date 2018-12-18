import math
import numpy as np
import matplotlib.pyplot as plt
import random
import chainer
import chainer.links as L 
import chainer.functions as F 
from chainer import Variable, optimizers, Chain

class FCNN(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.l1 = L.Linear(1,100)
			self.l2 = L.Linear(100,100)
			self.l3 = L.Linear(100,1)

	def __call__(self,x):
		h = F.sigmoid(self.l1(x))
		h = F.sigmoid(self.l2(h))
		h = self.l3(h)
		return h 

def generate_data(number_of_data,power):
	d1 = []
	d2 = []
	for i in range(number_of_data):
		val = random.uniform(0,10)
		d1.append([val])
		d2.append([val**(power)])

	return np.array(d1,dtype=np.float32),np.array(d2,dtype=np.float32) 

def main():
	power = 1/2 #This program learns y = x^power

	number_of_data = 1000
	max_epoch = 100
	batchsize = 50
	train_ratio = 0.8
	slice_position = int(number_of_data * train_ratio)

	data_input, data_output = generate_data(number_of_data,power)

	train_input = data_input[:slice_position]
	train_output = data_output[:slice_position]
	train_input = Variable(train_input)
	train_output = Variable(train_output)

	test_input = data_input[slice_position:]
	test_output = data_output[slice_position:]
	test_input = Variable(test_input)
	test_output = Variable(test_output)

	model = FCNN()
	optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
	optimizer.setup(model)

	#Train
	N = len(train_input)
	perm = np.random.permutation(N)
	for epoch in range(max_epoch):
		for i in range(0,N,batchsize):
			train_input_batch = train_input[perm[i:i + batchsize]]
			train_output_batch = train_output[perm[i:i + batchsize]]

			model.cleargrads()
			t = model(train_input_batch)
			loss = F.mean_squared_error(t, train_output_batch)
			loss.backward()
			optimizer.update()
		print("epoch:", epoch, "loss", loss.data)


		t = model(test_input)
		plt.scatter(test_input.array, test_output.data, label="Ground truth") #Ground Truth
		plt.scatter(test_input.array, t.array, label="Result of Learning") #Result 
		plt.xlabel("x", fontsize=16)
		plt.ylabel("y", fontsize=16)
		plt.xlim(0, 10)
		plt.ylim(0, 10**power)
		plt.legend(loc="lower right")
		plt.title("epoch{}".format(epoch), fontsize=16)
		plt.savefig("graph/epoch{}.png".format(epoch))
		plt.close()

	#Test
	loss = F.mean_squared_error(t, test_output)
	print("loss:", loss.data)

if __name__ == "__main__":
	main()
