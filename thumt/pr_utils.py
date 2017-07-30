import math
import numpy
import json
import cPickle
from mrt_utils import getRefDict, calBleu

def getPRBatch(x, xmask, y, ymask, config, model, data, fls):
	sampleN = config['sampleN_PR']
	myL = int(config['LenRatio_PR'] * len(y))
	samples, attn = model.sample(x.squeeze(), myL, sampleN)
	#attn.reshape(samples.shape)
	#print samples.shape
	#print 'attn:', attn.shape

	# format: {sentence:features(numpy array)}
	y_dic = getUnique(fls, samples, x, y, config, attn, model)

	Y, YM, features, ans = getYM(y_dic, y, config)
	features = numpy.array(features, dtype = 'float32')
	diffN = len(features)

	X = numpy.zeros((x.shape[0], diffN), dtype = 'int64')
	x = x + X
	X = numpy.zeros((x.shape[0], diffN), dtype = 'float32')
	xmask = xmask + X
	y = Y
	ymask = YM

	assert ans >= 0

	return x, xmask, y, ymask, features, ans

def getUnique(fls, samples, x, y, config, attn, model):
		
	dic = {}
	xn = x
	yn = y
	y = list(y.flatten())
	x = list(x.flatten())
	features = []

	# calculate feature for gold translation
	for fl in fls:
		if isinstance(fl, featureListAttn):
			attn_ans = model.get_attention(xn, numpy.ones(xn.shape, dtype = numpy.float32), yn, numpy.ones(yn.shape, dtype = numpy.float32))[0]
			add_info = [numpy.reshape(attn_ans, (attn_ans.shape[0], attn_ans.shape[1]))]
		else:
			add_info = None
		features.append(fl.getFeatures(x, cutSen(y, config), add_info = add_info))
	dic[json.dumps(cutSen(y, config))] = numpy.concatenate(features)
	
	# calculate features for samples
	for i in range(samples.shape[0]):
		tmp = list(samples[i])
		features = []
		for fl in fls:
			if isinstance(fl, featureListAttn):
				add_info = [attn[i]]
			else:
				add_info = None
			features.append(fl.getFeatures(x, cutSen(tmp, config), add_info = add_info))
		dic[json.dumps(cutSen(tmp, config))] = numpy.concatenate(features)

	return dic

def getYM(y_dic,truey,config):
	ans = -1
	y = [json.loads(i) for i in y_dic]
	truey = list(truey.flatten())
	n = len(y_dic)
	features = []
	max = 0 
	idx = 0
	# find the longest sentence and the index of gold translation
	for key in y_dic:
		tmp = json.loads(key)
		tmplen = len(tmp)
		if max < tmplen:
			max = tmplen
		if truey == tmp:
			ans = idx
		idx += 1
	Y = numpy.ones((max,n), dtype = 'int64') * config['index_eos_trg']
	Ymask = numpy.zeros((max, n), dtype = 'float32')
	i = 0
	for key in y_dic:
		features.append(y_dic[key])
		tmp = json.loads(key)
		ly = len(tmp)
		Y[0:ly,i] = numpy.asarray(tmp, dtype = 'int64')
		Ymask[0:ly, i] = 1
		i += 1
	return Y, Ymask, numpy.asarray(features, dtype = 'float32'), numpy.asarray(ans, dtype = 'int64')

def my_log(a):
	if a == 0:
		return -1000000
	return math.log(a)

def cutSen(x, config):
	if config['index_eos_trg'] not in x:
		return x
	else:
		return x[:x.index(config['index_eos_trg']) + 1]

class featureList(object):

	def __init__(self):
		pass

	def getScore(self, source, hypo, add_info = None):
		return (self.feature_weight * self.getFeatures(source, hypo, add_info)).sum()

class featureListRef(featureList):

	def __init__(self):
		pass

class featureListAttn(featureList):

	def __init__(self):
		pass

class feature_word(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		#load word table
		self.word_idx = {}
		self.word_s2t = []
		num_words = 0
		word_table = cPickle.load(open(config['word_table'], 'r'))
		writefile = open('word.txt', 'w')
		print 'total', len(word_table) ,'word entries'
		for i in word_table:
			if data.ivocab_src.has_key(i[0]) and data.ivocab_trg.has_key(i[1]):
				if self.word_idx.has_key(data.ivocab_src[i[0]]):
					self.word_idx[data.ivocab_src[i[0]]].append(num_words)
				else:
					self.word_idx[data.ivocab_src[i[0]]] = [num_words]
				self.word_s2t.append([data.ivocab_src[i[0]], data.ivocab_trg[i[1]]])
				num_words += 1
				print >> writefile, i[0] + ' ||| ' + i[1]
		print 'reserve', len(self.word_s2t), 'word features'
		self.feature_weight = numpy.ones((len(self.word_s2t),)) * config['feature_weight_word']

	def getFeatures(self, source, hypo, add_info = None):
		result = numpy.zeros((len(self.word_s2t),))
		for i in source:
			if self.word_idx.has_key(i):
				for j in self.word_idx[i]:
					if self.word_s2t[j][1] in hypo:
						result[j] = 1

		return result
	
class feature_phrase(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		#load phrase table
		self.phrase_idx = {}
		self.phrase_s2t = []
		num_phrases = 0
		phrase_table = cPickle.load(open(config['phrase_table'], 'r'))
		writefile = open('phrase.txt', 'w')
		print 'total', len(phrase_table) ,'phrase entries'
		for i in phrase_table:
			source_words = i[0].split(' ')
			target_words = i[1].split(' ')
			if len(source_words) > config['max_phrase_length'] or len(target_words) > config['max_phrase_length']:
				continue
			nounk = True
			for j in range(len(source_words)):
				if data.ivocab_src.has_key(source_words[j]):
					source_words[j] = data.ivocab_src[source_words[j]]
				else:
					nounk = False
			for j in range(len(target_words)):
				if data.ivocab_trg.has_key(target_words[j]):
					target_words[j] = data.ivocab_trg[target_words[j]]
				else:
					nounk = False
			if not nounk:
				continue
			phrase_source = ' '.join([str(k) for k in source_words])
			if self.phrase_idx.has_key(phrase_source):
				self.phrase_idx[phrase_source].append(num_phrases)
			else:
				self.phrase_idx[phrase_source] = [num_phrases] 
			self.phrase_s2t.append([phrase_source, ' '.join([str(k) for k in target_words])])
			num_phrases += 1
			print >> writefile, i[0]+' ||| '+i[1]
		print 'reserve', len(self.phrase_s2t), 'phrase features'
		self.feature_weight = numpy.ones((len(self.phrase_s2t),)) * config['feature_weight_phrase']

	def getFeatures(self, source, hypo, add_info = None):
		result = numpy.zeros((len(self.phrase_s2t),))
		phrase_hypo = ' '.join([str(k) for k in hypo])
		for i in range(len(source)):
			for j in range(i + 1,min(len(source),i + self.config['max_phrase_length'])):
				phrase_source = ' '.join([str(k) for k in source[i:j]])
				if self.phrase_idx.has_key(phrase_source):
					for k in self.phrase_idx[phrase_source]:
						if self.phrase_s2t[k][1] in phrase_hypo:
							result[k] = 1

		return result
	 
class feature_length(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_length_ratio']

	def getFeatures(self, x, y, add_info = None):
		if len(x) * self.config['length_ratio'] > len(y):
			return numpy.asarray([1.0 * len(y) / (len(x) * self.config['length_ratio'])], dtype = 'float32')
		else:
			return numpy.asarray([1.0 * (len(x) * self.config['length_ratio']) / len(y)], dtype = 'float32')

class feature_wordcount(featureList):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_wordcount']

	def getFeatures(self, x, y, add_info = None):
		return numpy.asarray([len(y)], dtype = 'float32')

def getMask(y,config):
	mask = numpy.ones((len(y),), dtype = 'float32')
	if config['index_eos_trg'] in y:
		mask[(y.index(config['index_eos_trg']) + 1):] = 0
	return mask

class feature_attention_coverage(featureListAttn):

	def __init__(self, config, data):
		self.config = config
		self.data = data
		self.feature_weight = numpy.ones((1,)) * config['feature_weight_attention_coverage']

	def getFeatures(self, x, y, add_info = None):
		#attention: len(y)*len(x)
		assert add_info
		print add_info[0].shape
		attention = add_info[0][:len(y)]
		#mask = getMask(y,self.config)
		#attn_sum = (attention*numpy.reshape(mask, (mask.shape[0],1))).sum(axis=0)
		attn_sum = attention.sum(axis = 0)
		attn_score = numpy.log(attn_sum.clip(0, 1)).sum()
		return numpy.asarray([attn_score], dtype = 'float32')

