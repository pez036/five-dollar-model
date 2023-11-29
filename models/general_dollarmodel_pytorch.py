import torch
from torch import nn, optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

seed = 7499629

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"**--Using {device}--**")

class imageDataSet(Dataset):

	def __init__(self, dataset):
		self.data = dataset

	def __len__(self):
		return len(self.data[0])

	def __getitem__(self,idx):
		return self.data[0][idx], self.data[1][idx]

class ResBlock(nn.Module):
	def __init__(self, kern_size=7, filter_count=128, upsampling=False):
		super().__init__()
		self.upsampling = upsampling
		self.kern_size = kern_size
		self.filter_count = filter_count
		self.layers = nn.Sequential(
			nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
			nn.ReLU(),
			nn.BatchNorm2d(self.filter_count),
			nn.Conv2d(self.filter_count, self.filter_count, kernel_size=self.kern_size, padding=3),
			nn.ReLU(),
			nn.BatchNorm2d(self.filter_count),
		)


	def forward(self, x):
		if self.upsampling:
			x = nn.Upsample(scale_factor=2, mode='nearest')(x)
		x1 = self.layers(x)
		return x1 + x

class Gen(nn.Module):
	def __init__(self, model_name, img_shape, lr, data_path, dataset_type, embedding_dim=384, z_dim=5, kern_size=7, filter_count=128, num_res_blocks=3, condition_type=''):
		super().__init__()

		self.embedding_dim = embedding_dim
		self.model_name = model_name
		self.img_shape = img_shape
		self.z_dim = z_dim
		self.filter_count = filter_count
		self.kern_size = kern_size
		self.num_res_blocks = num_res_blocks

		self.lr = lr
		self.data_path = data_path
		self.dataset_type = dataset_type
		self.condition_type = condition_type

		self.lin1 = nn.Linear(self.embedding_dim + self.z_dim, self.filter_count * 4 * 4)

		self.res_blocks = nn.Sequential()
		for i in range(self.num_res_blocks):
			self.res_blocks.append(ResBlock(self.kern_size, self.filter_count, i < 2))

		self.padding = nn.ZeroPad2d(1)
		self.last_conv = nn.Conv2d(in_channels=self.filter_count, out_channels=16, kernel_size=9)
		self.softmax = nn.Softmax(dim=1)


	def forward(self, embedding, z_dim):
		enc_in_concat = torch.cat((embedding, z_dim), 1)
		x = self.lin1(enc_in_concat)
		x = x.view(-1, self.filter_count, 4, 4)
		# x = torch.reshape(x, (4,4,self.filter_count))
		x = self.res_blocks(x)
		x = self.padding(x)
		x = self.last_conv(x)
		return self.softmax(x)


def load_data(path, scaling_factor=6, batch_size=256):
	data = np.load(path, allow_pickle=True).item()
	images = np.array(data['images'])
	labels = data['labels']

	embeddings = data['embeddings']
	if isinstance(embeddings, list):
		embeddings = np.array(embeddings)
	embeddings = embeddings * scaling_factor

	images, images_test, labels, labels_test, embeddings, embeddings_test = train_test_split(
	images, labels, embeddings, test_size=24, random_state=seed)

	train_dataset = [embeddings, images]
	test_dataset = [embeddings_test, images_test]

	train_set = DataLoader(imageDataSet(train_dataset),
					   batch_size=batch_size,
					   shuffle=True,
					   num_workers= 8 if device == 'cuda' else 1,
					   pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

	test_set = DataLoader(imageDataSet(test_dataset),
					  batch_size=batch_size,
					  shuffle=True,
					  num_workers= 8 if device == 'cuda' else 1,
					  pin_memory=(device=="cuda")) # Makes transfer from the CPU to GPU faster

	return train_set, test_set

def train(model, EPOCHS, batch_size):

	#train_set, test_set = load_data("../datasets/maps_gpt4_aug.npy")
	train_set, test_set = load_data(model.data_path, batch_size)

	loss_metric_train = torch.zeros(EPOCHS).to(device)

	model.to(device)

	optimizer = optim.Adam(model.parameters())

	for epoch in range(EPOCHS):

		for embeddings, ytrue in train_set:

			optimizer.zero_grad()
			outputs = model(embeddings.to(device), torch.rand(len(embeddings), 5).to(device))
			loss = nn.NLLLoss()(torch.log(outputs), ytrue.argmax(dim=-1).to(device))

			loss_metric_train[epoch] += loss

			loss.backward()
			optimizer.step()

		print(f"Epoch:{epoch} -- Loss:{loss_metric_train[epoch]:.2f}")

#BATCH_SIZE = 256
_input_shape = (10, 10, 16)
_data_path = "../datasets/maps_gpt4_aug.npy"
_lr = 0.005
_model_name = "map_model"
_dataset_type = 'map'
_embedding_dim = 384
_z_dim = 5
_kern_size = 7
_filter_count = 128
_num_res_blocks = 3

_epochs = 100
_batch_size = 256

mapmodel = Gen(model_name=_model_name, img_shape=_input_shape, lr=_lr, data_path=_data_path, dataset_type=_dataset_type, 
				embedding_dim=_embedding_dim, z_dim=_z_dim, kern_size=_kern_size, filter_count=_filter_count, num_res_blocks=_num_res_blocks)
#(model_name="test_modelname", img_shape=input_shape, lr=0.0005, embedding_dim=384, z_dim=5, filter_count=128, kern_size=5, num_res_blocks=3, dataset_type='map', data_path='datasets/maps_noaug.npy')
train(mapmodel, _epochs, _batch_size)