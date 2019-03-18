import argparse
import random
import numpy as np

import networkx as nx	

import pickle
import os

from resnet import *

def generate_dataset(args):
	dataset_path = "dataset_%d_%.2f_%d_%d.pickle"%(args.num_vertices, 100.0*args.sparsity, 
					args.train_samples, args.test_samples)

	if not os.path.isfile(dataset_path):
		# Dataset
		train_data_in = np.zeros((args.train_samples, args.num_vertices, args.num_vertices))
		train_data_out = np.zeros((args.train_samples, args.num_vertices))

		test_data_in = np.zeros((args.test_samples, args.num_vertices, args.num_vertices))
		test_data_out = np.zeros((args.test_samples, args.num_vertices))

		print("Generating training dataset")
		quota_id = 0
		completed_perc = 0
		for id in range(args.train_samples):
			graph =	nx.erdos_renyi_graph(args.num_vertices, args.sparsity)

			# Input
			for edge in graph.edges():
				train_data_in[id, edge[0], edge[1]] = 1
				train_data_in[id, edge[1], edge[0]] = 1

			# Output
			cc = list(nx.closeness_centrality(graph).values())
			train_data_out[id,:] = cc

			quota_id += 1
			if (quota_id/args.train_samples) >= 0.1:
				quota_id = 0
				completed_perc += 10
				print("Completed  %.2f/100"%(completed_perc))

		print("Generating test dataset")
		quota_id = 0
		completed_perc = 0
		for id in range(args.test_samples):
			graph =	nx.erdos_renyi_graph(args.num_vertices, args.sparsity)

			# Input
			for edge in graph.edges():
				test_data_in[id, edge[0], edge[1]] = 1
				test_data_in[id, edge[1], edge[0]] = 1

			# Output
			cc = list(nx.closeness_centrality(graph).values())
			test_data_out[id,:] = cc

			quota_id += 1
			if (quota_id/args.test_samples) >= 0.1:
				quota_id = 0
				completed_perc += 10
				print("Completed %.2f/100"%(completed_perc))


		dataset = {"train":[train_data_in, train_data_out], 
					"test": [test_data_in, test_data_out]}

		# Dumping to a pickel
		dataset_fh = open(dataset_path, "wb")
		pickle.dump(dataset, dataset_fh)
		dataset_fh.close()

	else:
		dataset_fh = open(dataset_path, "rb")
		dataset = pickle.load(dataset_fh)
		dataset_fh.close()

	return dataset


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--num-vertices", type=int, default=32)
	parser.add_argument("--sparsity", type=float, default=0.25)
	parser.add_argument("--train-samples", type=int, default=128000)
	parser.add_argument("--test-samples", type=int, default=12800)

	parser.add_argument('--epochs', default=32, type=int, metavar='N',
					help='number of total epochs to run')
	parser.add_argument('-b', '--batch-size', default=256, type=int,
						metavar='N',
						help='mini-batch size (default: 256), this is the total '
							 'batch size of all GPUs on the current node when '
							 'using Data Parallel or Distributed Data Parallel')
	parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
						metavar='LR', help='initial learning rate', dest='lr')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)',
						dest='weight_decay')

	parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')

	args = parser.parse_args()

	# Generate dataset
	dataset = generate_dataset(args)

	# model
	net = ResNet18(num_classes=args.num_vertices)
	net = net.cuda()

	# Loss function
	criterion = nn.MSELoss().cuda()
	l1_criterion = nn.L1Loss().cuda()

	# Optimizer
	optimizer = torch.optim.SGD(net.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay,
								nesterov=True)


	portions = [1/2, 3/4]
	milestones = [int(portion*args.epochs) for portion in portions]
	print(milestones)
	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)


	for epoch in range(args.epochs):  # loop over the dataset multiple times
		lr_scheduler.step() # Adjust learning rate

		num_train_iters = args.train_samples // args.batch_size
		running_loss = 0
		for iter_id in range(num_train_iters):
			inputs  = dataset["train"][0][iter_id*args.batch_size : (iter_id+1)*args.batch_size]
			outputs = dataset["train"][1][iter_id*args.batch_size : (iter_id+1)*args.batch_size]

			# Adding channel dimension
			inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
			inputs, outputs = np.float32(inputs), np.float32(outputs)

			inputs, outputs = torch.from_numpy(inputs) , torch.from_numpy(outputs)
			inputs, outputs = inputs.cuda(), outputs.cuda()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			p_outputs = net(inputs)
			loss = criterion(p_outputs, outputs)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if iter_id % args.print_freq == args.print_freq-1:    # print every 128 mini-batches
				print('[Epoch %d, %5d/%d] loss: %f' %
					  (epoch + 1, iter_id + 1, num_train_iters, running_loss / args.print_freq))
				running_loss = 0.0


		# Validation
		with torch.no_grad():
			num_test_iters = args.test_samples // args.batch_size
			test_loss = 0
			test_l1_loss = 0
			for iter_id in range(num_test_iters):
				inputs  = dataset["test"][0][iter_id*args.batch_size : (iter_id+1)*args.batch_size]
				outputs = dataset["test"][1][iter_id*args.batch_size : (iter_id+1)*args.batch_size]

				# Adding channel dimension
				inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
				inputs, outputs = np.float32(inputs), np.float32(outputs)

				inputs, outputs = torch.from_numpy(inputs) , torch.from_numpy(outputs)
				inputs, outputs = inputs.cuda(), outputs.cuda()

				p_outputs = net(inputs)
				loss = criterion(p_outputs, outputs)
				test_loss += loss.item()

				# L1 loss
				l1_loss = l1_criterion(p_outputs, outputs)
				test_l1_loss += l1_loss.item()


				if iter_id == 0:
					for i in range(1):
						print("Test case ",i)
						print(outputs[i])
						print(p_outputs[i])
						cur_loss = l1_criterion(p_outputs[i], outputs[i])
						print("Loss", cur_loss)
						print()

			print("Test loss(L2): ", test_loss/num_test_iters)
			print("Test loss(L1): ", test_l1_loss/num_test_iters)

	
	
	
