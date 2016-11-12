中文手写地址识别流程
	1.训练
		1.训练数据集
			1.关键字训练集（15class）
				1.生成训练集
					
			2.单字训练集（3755class）
			3.加入direct的关键字训练集（15class）
			4.加入direct的单子训练集（3755class）
		2.训练流程
			A.关键字训练（15class）
				从hcl数据集中选出14+1字，调整图像大小和其他相关参数，训练之。
					省，市，县，区，乡，村，镇，路，巷，弄，街，号，栋，楼，X
				1.
				
			B.单字训练（3755class）
				Hcl数据集大规模单字训练。
				得出的结果是一个列表而不是一位热键，列表从前往后说明了该字符matrix的最可能值
				Want:[32,158,2,59,68,…n](n<=10)
				Not:[0,0,0,…,1,…,0](3755个)
			C.对A加入direct作为非binary图像，而是0，1，2，3，4值
			D.单字direct，B加入direct
	2.识别
		1.输入一个binary的图像img
		2.对img求得其竖向的统计col_histogram
		3.由col_histogram分割图像为char_list，train_result_list = char_list, default=[char, []]
		4.初始化地址层次树addr_tree
		5.维护一个识别结果path_list，也是一个addr_tree的一条路径，加入root“中国”，即["中国"]
		6.for i in char_list:
			先做关键字识别得到分类1~15
			If I is a keyword（key_word_result（i）！=15）:
				Train_result_list[i][1]=key_word_result(i)
			Else:
				Train_result_list[i][1]=single_char_result(i)，是一个list
		7.由list_quora算法根据addr_tree和path_list加入train_result_list作为参数得出result
		8.输出结果
	2.测试
		测试数据集
			1.关键字测试集（15class）
				测试关键字识别准确率
			2.单字测试集（3755class）
				单子识别准确率
			3.加入direct的关键字测试集（15class）
				。。。
			4.加入direct的单子测试集（3755class）
				。。。
			5.完整的单行binary地址图像png
				
			5.完整的单行direct地址图像png
				
	3.难点
		1.如何在训练中不得出one_hot的输出结果而是得到candidate_list
		2.如何实现list_quora算法
		3.如何整合direct到binary中去
