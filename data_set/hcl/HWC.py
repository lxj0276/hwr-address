#!/usr/bin/env python
#-*- coding: utf8 -*-
# a class to return bin handwriting character matrix(es).
import os

THIS_DIR = os.path.split(os.path.realpath(__file__))[0]
SUB_DIR = '/hcl_data/'

def __init__():
	pass

class HWC():
	def __init__(self,index='hh001'):
		filename = THIS_DIR + SUB_DIR + index + '.hcl'
		self.filename=filename
		self.matrixlist=[]
		self.matrixlist=self.fetchall()
		self._fetchnum=0
	def fetchall(self):
		if len(self.matrixlist)==0:
			matrixlist=[]
			with open(self.filename,'rb') as f:
				r=f.read()
			for i in range(len(r)//512):
				matrixlist.append(HWC._decode2matrix(r[i*512:i*512+512]))
			return matrixlist
		else:
			return self.matrixlist

	def fetchone(self):
		self._fetchnum+=1
		return self.matrixlist[self._fetchnum]
	
	@staticmethod
	def _decode2matrix(string):
		sx=[ord(i) for i in string]
		s=[[0 for j in range(64)]for i in range(64)]
		for i in range(64):
			for k in range(8):
				dat=sx[i*8+k]
				for j in range(8):
					s[i][k*8+j]=(dat>>(7-j))&0x01
		return s
	
	@staticmethod
	def show(m):
		for i in m:
			s=''.join(list(map(str,i)))
			print s

	def getone(self,index):
		return self.matrixlist[index]

def main():
	hwc=HWC()
	c=hwc.getone(23)
	for i in c:
		s=''.join(list(map(str,i)))
		print s

if __name__=='__main__':
	main()
