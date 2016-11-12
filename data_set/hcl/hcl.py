# hcl dataset.
# coding: utf-8
from __future__ import division
import os
import random

THIS_DIR = os.path.split(os.path.realpath(__file__))[0]+'/'
SUB_DIR = 'hcl'
DECODED_DIR = 'decoded'
char_dict_filename = 'GB2312_3755.txt'

def __init__():
	pass

class input_data:
	hcl_pickle_path = os.path.join(THIS_DIR, SUB_DIR, 'hcl.dmp')
	
	def __init__(self, choosing_characters = ['中', '国'], num_of_every_train_sample = 500, num_of_every_test_sample = 100, raw_data = True, one_hot = True, size = (64, 64), direct_info = True):
		index_of_choosing = input_data.choosing_characters_list_to_index(choosing_characters)
		self.train = train(index_of_choosing, num_of_every_train_sample, raw_data, one_hot, size, direct_info)
		self.test = test(index_of_choosing, num_of_every_test_sample, raw_data, one_hot, size, direct_info)

	@staticmethod
	def choosing_characters_list_to_index(choosing_characters):
		if choosing_characters[0] == 1:
			return choosing_characters[1:]
		else:
			with open(THIS_DIR + char_dict_filename, 'r') as f:
				char_dict = list(map(lambda x: x.replace('\n', ''), f.readlines()))
				index_of_choosing = []
				for i in choosing_characters:
					try:
						index_of_choosing.append(char_dict.index(i))
					except:
						index_of_choosing.append(-1)
				return index_of_choosing

	@staticmethod
	def read_data(index_of_choosing, num_of_every_sample, raw_data, one_hot, size, direct_info, type_of_want, num_of_want ):
		file_path = os.path.join(THIS_DIR, SUB_DIR, type_of_want, DECODED_DIR)
		decoded = os.path.exists(file_path)
		char_size = 64*64
		char_size_root = 64
		if not decoded:
			file_path = os.path.split(file_path)[0]
			char_size = 512
		char_list = []
		label_list = []
		original_index = index_of_choosing
		if num_of_want != len(original_index):
			index_of_choosing = [index_of_choosing[random.randrange(len(index_of_choosing))] for i in range(num_of_want)]
		for i in index_of_choosing:
			#print '%d th sample' % i
			k = i							# 这三行是为了适应兜底项
			if k == -1:						# 一般应该只有一项
				k = random.randrange(3755)	# 兜底项随机选取，标签为-1
			filename = os.path.join(file_path, str(k + 1) + '.chars') #此处加一是因为hcl数据集命名从1开始
			with open(filename, "rb") as f:
				r = f.read()[:num_of_every_sample*char_size]
				for j in range(num_of_every_sample):
					if not decoded:
						character = input_data.c_bytes_decode(r[j*char_size:j*char_size+char_size])
					else:
						rr = r[j*char_size:j*char_size+char_size]
						character = [[ord(rr[k*char_size_root+l]) for l in range(char_size_root)] for k in range(char_size_root)]
						
					if size != (64, 64):
						character = input_data.resize(character, size)
					if direct_info:
						character = input_data.direct_info(character)
					if raw_data:
						character = reduce(lambda x,y: x+y, character)
					char_list.append(character)
					
					if one_hot:
						label = [0 for x in range(len(original_index))]
						label[original_index.index(i)] = 1
						label_list.append(label)
					else:
						label_list.append(i)
		return input_data.random_order(char_list, label_list)

	@staticmethod
	def random_order(char_list, label_list):
		l = len(char_list)
		x = []
		y = []
		for i in range(l):
			r = random.randrange(l)
			x.append(char_list[r])
			y.append(label_list[r])
		return [x,y]
		
	@staticmethod
	def c_bytes_decode(string):
		sx=[ord(i) for i in string]
		s=[[0 for j in range(64)]for i in range(64)]
		for i in range(64):
			for k in range(8):
				dat=sx[i*8+k]
				for j in range(8):
					s[i][k*8+j]=(dat>>(7-j))&0x01
		return s

	@staticmethod
	def show(char_matrix):
		for i in char_matrix:
			s=''.join(list(map(str,i)))
			print s

	@staticmethod
	def resize(character, size):
		s_size = (len(character), len(character[0]))
		a,b = size[0]/s_size[0],size[1]/s_size[1]
		if a==b==1:
			return character
		character = [[character[int(i/a)][int(j/b)] for j in range(size[1])]for i in range(size[0])]
		return character
		
	@staticmethod
	def direct_info(character):
		#TODO: finish this function.
		#to generate a 8xsize div tensor to get character's direction information
		def direct(i,j,character,size):
			d1 = lambda i,j:sum(character[i])
			d2 = lambda i,j:sum([character[x][x-abs(i-j)] for x in range(abs(i-j),size[0])])
			d3 = lambda i,j:sum([character[x][j] for x in range(size[0])])
			d4 = lambda i,j:sum([character[y][i+j-y] for y in range(max((0,i+j-size[1]+1)),min((size[1],i+j+1)))])
			x = [d1(i,j),d2(i,j),d3(i,j),d4(i,j)]
			return x.index(max(x)) + 1
		size = (len(character), len(character[0]))
		x = [[0 for j in range(size[1])] for i in range(size[0])]
		for i in range(size[0]):
			for j in range(size[1]):
				if character[i][j] == 1:
					#character[i][j] = 0
					x[i][j] = direct(i,j,character, size)
		character = x
		return character

	
class train:
	def __init__(self, index_of_choosing, num_of_every_train_sample, raw_data, one_hot, size, direct_info):
		self.index = 0
		self.index_of_choosing = index_of_choosing
		self.num_of_every_train_sample = num_of_every_train_sample
		self.raw_data = raw_data
		self.one_hot = one_hot
		self.size = size
		self.direct_info = direct_info
		
	def next_batch(self, batch_size):
		if (self.index + batch_size <= self.size):
			self.index += batch_size
		else:
			self.index = 50
		return input_data.read_data(self.index_of_choosing, self.num_of_every_train_sample, self.raw_data, self.one_hot, self.size, self.direct_info, 'train', batch_size)
		
class test:
	def __init__(self, index_of_choosing, num_of_every_test_sample, raw_data, one_hot, size, direct_info):
		self.index = 0
		self.index_of_choosing = index_of_choosing
		self.num_of_every_test_sample = num_of_every_test_sample
		self.raw_data = raw_data
		self.one_hot = one_hot
		self.size = size
		self.direct_info = direct_info
		
	def next_batch(self, batch_size):
		if (self.index + batch_size <= self.size):
			self.index += batch_size
		else:
			self.index = 50
		return input_data.read_data(self.index_of_choosing, self.num_of_every_test_sample, self.raw_data, self.one_hot, self.size, self.direct_info, 'test', batch_size)
		
def main():
	#hcl = input_data(['省','市','县','区','乡','镇','村','巷','弄','街'], 5, 2, False, True, (28, 28), True)
	hcl = input_data([1]+[i for i in range(3755)], 500, 200, False, True, (28, 28), False)
	input_data.show(hcl.train.next_batch(7)[0][2])
	
	
	
if __name__ == '__main__':
	main()
