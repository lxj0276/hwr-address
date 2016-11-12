# coding: utf-8
# recognize.py

from preprocessing import preprocess

def __init__():
	pass

def recognize(img):
	charlist = preprocess(img)
	keyword_charlist = keyword_recognizer(charlist)
	keyword_province_charlist = province_recognizer(keyword_charlist)
	preresult = validation_recognizer(keyword_province_charlist)
	detail_result = detail_recognizer(preresult)
	result = checkall(detail_result)
	return result