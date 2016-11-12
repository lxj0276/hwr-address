r = [1,2,3,4,5,6,7,8,9]
char_size_root = 3
print [[r[i*char_size_root+j] for j in range(char_size_root)] for i in range(char_size_root)]