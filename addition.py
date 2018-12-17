import numpy as np
import random
import chainer
import chainer.links as L 
import chainer.functions as F 
from chainer import Variable, optimizers, Chain, datasets

class FCNN(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.l1 = L.Linear(2,50)
			self.l2 = L.Linear(50,50)
			self.l3 = L.Linear(50,1)

	def __call__(self,x):
		h = F.relu(self.l1(x))
		h = F.relu(self.l2(h))
		h = self.l3(h)
		return h 

def generate_data(number_of_data):
	d1 = []
	d2 = []
	for i in range(number_of_data):
		val1 = random.uniform(-50,50)
		val2 = random.uniform(-50,50)
		d1.append([val1, val2])
		d2.append([val1+val2])

	return np.array(d1,dtype=np.float32),np.array(d2,dtype=np.float32) 

def main():
	number_of_data = 1000
	max_epoch = 20
	batchsize = 100
	train_ratio = 0.8
	slice_position = int(number_of_data * train_ratio)

	data_input, data_output = generate_data(number_of_data)

	train_input = data_input[:slice_position]
	train_output = data_output[:slice_position]
	train_input = Variable(train_input)
	train_output = Variable(train_output)

	test_input = data_input[slice_position:]
	test_output = data_output[slice_position:]
	test_input = Variable(test_input)
	test_output = Variable(test_output)

	model = FCNN()
	optimizer = optimizers.MomentumSGD(lr=0.0001, momentum=0.9)
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
			loss = F.mean_squared_error(t,train_output_batch)
			loss.backward()
			optimizer.update()
		print("epoch:",epoch,"loss",loss.data)

	#Test
	print(model(Variable(np.array([[14,-2]],dtype=np.float32))).array)
	t = model(test_input)
	loss = F.mean_squared_error(t,test_output)
	print("loss:",loss.data)

if __name__ == "__main__":
	main()
