������д��ַʶ������
	1.ѵ��
		1.ѵ�����ݼ�
			1.�ؼ���ѵ������15class��
				1.����ѵ����
					
			2.����ѵ������3755class��
			3.����direct�Ĺؼ���ѵ������15class��
			4.����direct�ĵ���ѵ������3755class��
		2.ѵ������
			A.�ؼ���ѵ����15class��
				��hcl���ݼ���ѡ��14+1�֣�����ͼ���С��������ز�����ѵ��֮��
					ʡ���У��أ������磬�壬��·���Ū���֣��ţ�����¥��X
				1.
				
			B.����ѵ����3755class��
				Hcl���ݼ����ģ����ѵ����
				�ó��Ľ����һ���б������һλ�ȼ����б��ǰ����˵���˸��ַ�matrix�������ֵ
				Want:[32,158,2,59,68,��n](n<=10)
				Not:[0,0,0,��,1,��,0](3755��)
			C.��A����direct��Ϊ��binaryͼ�񣬶���0��1��2��3��4ֵ
			D.����direct��B����direct
	2.ʶ��
		1.����һ��binary��ͼ��img
		2.��img����������ͳ��col_histogram
		3.��col_histogram�ָ�ͼ��Ϊchar_list��train_result_list = char_list, default=[char, []]
		4.��ʼ����ַ�����addr_tree
		5.ά��һ��ʶ����path_list��Ҳ��һ��addr_tree��һ��·��������root���й�������["�й�"]
		6.for i in char_list:
			�����ؼ���ʶ��õ�����1~15
			If I is a keyword��key_word_result��i����=15��:
				Train_result_list[i][1]=key_word_result(i)
			Else:
				Train_result_list[i][1]=single_char_result(i)����һ��list
		7.��list_quora�㷨����addr_tree��path_list����train_result_list��Ϊ�����ó�result
		8.������
	2.����
		�������ݼ�
			1.�ؼ��ֲ��Լ���15class��
				���Թؼ���ʶ��׼ȷ��
			2.���ֲ��Լ���3755class��
				����ʶ��׼ȷ��
			3.����direct�Ĺؼ��ֲ��Լ���15class��
				������
			4.����direct�ĵ��Ӳ��Լ���3755class��
				������
			5.�����ĵ���binary��ַͼ��png
				
			5.�����ĵ���direct��ַͼ��png
				
	3.�ѵ�
		1.�����ѵ���в��ó�one_hot�����������ǵõ�candidate_list
		2.���ʵ��list_quora�㷨
		3.�������direct��binary��ȥ
