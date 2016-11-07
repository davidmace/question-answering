# Notes:
# Lose 2.4% of questions because the entids are not in feebase-ents.txt


from pyspark import SparkContext, SparkConf
import operator
import re, string
from nltk import ngrams
from collections import defaultdict
import operator as op
import math
import numpy as np

########################################################################
### Global helpers
########################################################################

def outer_join_dicts(f,d1,d2) :
	d3 = {}
	for k in set.union(set(d1.keys()),set(d2.keys())) :
		d3[k] = f( d1[k] if k in d1 else 0 , d2[k] if k in d2 else 0 )
	return d3

def map_values(f,d) :
	return dict(map(lambda (k,v): (k, f(v)), d.iteritems()))

def remove_double_spaces(s) :
	return ' '.join(filter(lambda x: x!='', s.split(' ')))

def group_pairs_by_first(pairs) :
	d = defaultdict(list)
	for pair in pairs :
		d[pair[0]].append(pair[1])
	return d


########################################################################
### Load unigram counts
########################################################################

def load_unigram_cts() :
	with open('unigrams.txt','r') as f :
		lines = f.read().split('\n')
		word_cts = defaultdict(int)
	for line in lines :
		parts = line.split('\t')
		freq = int(parts[0])
		word = re.sub('[^a-z\'\-]+', '', parts[1].strip().lower())
		word_cts[word] += freq
	doclogfreq = defaultdict(float)
	for word in word_cts :
		doclogfreq[word] = math.log(word_cts[word])

	with open('filter-words-100.txt','r') as f :
		filter_words = set(f.read().split('\n'))

	return (doclogfreq, filter_words)


########################################################################
### Store entity id to name mapping
########################################################################

def make_uid_name_pair(line) :
	name = line[:line.find('\t')]
	if '(' in name :
		name = name[:name.find('(')]
	name = re.sub('[^a-zA-Z0-9\' \-]+', '', name).lower().strip()
	uid = line[line.rfind('/'):line.rfind('>')]
	if ' ' in uid or len(uid)>10 :
		uid = ''
		name = ''
	return (uid,name)

def process_entity_file(sc) :
	sc.textFile("freebase-ents.txt").map(make_uid_name_pair)\
	.reduceByKey(lambda a,b: a).coalesce(1).saveAsSequenceFile("entid2name")


###########################################################################
### Make list of all entity ids that we need to exact match
###########################################################################

def get_all_ids(line) :
	parts = line.split('\t')
	uid1 = parts[0]
	uid1 = uid1[uid1.rfind('/'):]
	reltype = parts[1]
	reltype = reltype.replace('www.freebase.com','')
	l = []
	l.append(uid1)
	for i in range(2,len(parts)) : # can be multiple direct objects in relationship
		uid2 = parts[i]
		uid2 = uid2[uid2.rfind('/'):]
		l.append(uid2)
	return l

def process_entid_list(sc) :
	sc.textFile("freebase-rules.txt").map(get_all_ids).flatMap(lambda x: x)\
		.distinct().coalesce(1).saveAsTextFile("present-entids2")


###########################################################################
### Make reduced entid to name map that only has entities present in the ruleset
###########################################################################

def process_entname_list(sc) :
	present_entids = sc.textFile('present-entids/part-00000')
	entid2name = sc.sequenceFile('entid2name/part-00000')
	present_id_map = present_entids.map(lambda x: (x,1))
	entid2name.join(present_id_map).coalesce(1).saveAsTextFile("entid2name-important")

def load_ent2name() :
	with open('entid2name-important/part-00000','r') as f :
		lines = f.read().split('\n')
		pair_list = [(line[3:line.find('\',')],remove_double_spaces(line[line.find(' (u')+4:-6])) for line in lines]
		entid2name_important = dict( pair_list )
		entname2id_important = defaultdict(list)
		for id in entid2name_important :
			entname2id_important[ entid2name_important[id] ].append(id)
		entname_set = set( [tuple(s.split()) for s in entid2name_important.values()] )
	return (entid2name_important, entname2id_important, entname_set)


###########################################################################
### Load rules into memory
###########################################################################

def process_rule_line(line) :
	parts = line.split('\t')
	uid1 = parts[0]
	uid1 = uid1[uid1.rfind('/'):]
	reltype = parts[1]
	reltype = reltype.replace('www.freebase.com','')
	#l = []
	#for i in range(2,len(parts)) : # can be multiple direct objects in relationship
	#	uid2 = parts[i]
	#	uid2 = uid2[uid2.rfind('/'):]
	#	l.append((uid1,(reltype,uid2)))
	return (uid1,reltype)

def process_rules(sc) :
	sc.textFile("freebase-rules.txt").map(process_rule_line).distinct().coalesce(1).saveAsTextFile("rules")

def load_rules() :
	rules = defaultdict(list)
	with open('rules/part-00000','r') as f :
		lines = f.read().split('\n')
		for line in lines :
			parts = line.split(',')
			id = parts[0][3:-1]
			rel = parts[1][3:-2]
			rules[id].append(rel)
	return rules










