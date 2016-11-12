# coding: utf-8
# preprocessing.py

'''
to preprocess a hw picture to a list of char matrix
input: picture
output: charlist
'''

from PIL import Image

def __init__():
	pass

def preprocess(img, wanna_size):
	matrix = img_to_matrix(img)
	rows = matrix_to_rows(matrix)
	charlist = rows_to_charlist(rows)
	charlist = resize_all(charlist, wanna_size)
	return charlist

def img_to_matrix(img):
	if img.mode != '1':
		img = img.convert('1')
	d = img.getdata()
	size = img.size
	m = [[d[j*size[1]+size[0]] for j in range(size[1])] for i in range(size[0])]
	return m

def matrix_to_rows(matrix):
	x = list(map(sum, matrix))
	y = [0] + [i if (x[i-1]>x[i])&(x[i+1]>x[i]) else 0 for i in range(1, len(x)-1)] + [-1]
	rows = [matrix[y[i]:y[i+1]] for i in range(len(y)-1)]
	return rows

def rows_to_charlist(rows):
	'''the way same as above'''
	pass
	
	
def resize_all(charlist, size):
	def resize(character, size):
		s_size = (len(character), len(character[0]))
		a,b = size[0]/s_size[0],size[1]/s_size[1]
		if a==b==1:
			return character
		character = [[character[int(i/a)][int(j/b)] for j in range(size[1])]for i in range(size[0])]
		return character
	return [resize(i, size) for i in charlist]
