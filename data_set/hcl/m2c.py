# to change hcl storage from manbased to charbased.

def read_a_file_to_str(filename):
	with open(filename, "rb") as f:
		r = f.read()
		return r

def read_ith_char_from_str(string, i, length):
	return string[i*length:i*length+length]

def write_str_to_a_file(filename, string):
	with open(filename, "ab") as f:
		f.write(string)
		return True

def loop_for_all_file(filedir,outputdir, filenum, type_of_file):
	for i in range(1, filenum + 1):
		filename = filedir + type_of_file + '{:0>3}'.format(i) + '.hcl'
		string = read_a_file_to_str(filename)
		for j in range(1, 3756):
			write_to_filename = outputdir + str(j) + '.chars'
			length = 512
			s = read_ith_char_from_str(string, j, length)
			write_str_to_a_file(write_to_filename, s)
		print('finished with', str(i))
	return True

def c_bytes_decode(string):
	sx=[ord(i) for i in string]
	s=[[0 for j in range(64)]for i in range(64)]
	for i in range(64):
		for k in range(8):
			dat=sx[i*8+k]
			for j in range(8):
				s[i][k*8+j]=(dat>>(7-j))&0x01
	return s

def loop_file(filedir, subdir):
	for i in range(1, 3756):
		filename = filedir + str(i) + '.chars'
		s = read_a_file_to_str(filename)
		ss = ''
		for j in range(len(s)/512):
			x = s[j*512:j*512+512]
			x = c_bytes_decode(x)
			x = reduce(lambda x,y: x+y, x)
			x = ''.join([chr(k) for k in x])
			ss+=x
		subfilename = filedir + subdir + str(i) + '.chars'
		write_str_to_a_file(subfilename, ss)
		print '%d th finished!' %i
	
def main():
	#loop_for_all_file('hcl_data/','hcl/test/', 300, 'hh')
	#loop_for_all_file('hcl_data/','hcl/train/', 700, 'xx')
	loop_file('hcl/test/', 'decoded/')
	loop_file('hcl/train/', 'decoded/')

if __name__ == '__main__':
	main()
