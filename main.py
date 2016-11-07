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

def flatten_one_layer(l) :
    return [item for sublist in l for item in sublist]


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

# www.freebase.com/m/03_7vl	www.freebase.com/people/person/profession	www.freebase.com/m/09jwl 
def process_rule_line(line) :
	parts = line.split('\t')
	uid1 = parts[0]
	uid1 = uid1[uid1.rfind('/'):]
	reltype = parts[1]
	reltype = reltype.replace('www.freebase.com','')
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


###########################################################################
### Find possible mispellings by method from http://emnlp2014.org/papers/pdf/EMNLP2014171.pdf
###########################################################################

def make_mispelling_resources(entname_set) :

	# map letters to prime numbers
	primes_letters = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
	primes_numbers = [103,109,113,127,131,137,139,149,151,157]
	primes_all = primes_letters + primes_numbers + [163,167,173]
	primes_map = {' ':163,'-':167,'\'':173}
	for i in range(26) :
		primes_map[chr(ord('a')+i)] = primes_letters[i]
	for i in range(10) :
		primes_map[chr(ord('0')+i)] = primes_numbers[i]

	# list of factors that entity letter score can be off by for one or two errors
	possible_spelling_ratios = set( flatten_one_layer([[1.0*x*y,1.0*x/y,1.0*y/x,1.0/x/y] for x in primes_all for y in primes_all])
				+ flatten_one_layer([[1.0*x,1.0/x] for x in primes_all]) )

	# map of spelling score to entity
	ent_spell_scores = {}
	for ent in entname_set :
		num_list = [primes_map[c] for c in ' '.join(ent)]
		if len(num_list)==0 or len(num_list)>40 :
			continue
		ent_spell_scores[float(reduce(op.mul,num_list))] = ent

	return (primes_map, ent_spell_scores, possible_spelling_ratios)

def edit_distance(s1, s2):
	if len(s1) > len(s2):
		s1, s2 = s2, s1
	distances = range(len(s1) + 1)
	for i2, c2 in enumerate(s2):
		distances_ = [i2+1]
		for i1, c1 in enumerate(s1):
			if c1 == c2:
				distances_.append(distances[i1])
			else:
				distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
		distances = distances_
	return distances[-1]

# return list of entities off by 1 or 2 letters from ent
def find_mispellings(ent, primes_map, ent_spell_scores, possible_spelling_ratios) :

	# look through 300 values of possible entity spelling scores
	find_val = reduce(op.mul,[primes_map[c] for c in ' '.join(ent)])
	possibilities = []
	for ratio in possible_spelling_ratios :
		if find_val*ratio in ent_spell_scores :
			possibilities.append(ent_spell_scores[long(find_val*ratio)])

	# use expensive edit distance method on reduced list to account for letter order
	found = []
	for poss in possibilities :
		if edit_distance(' '.join(poss),' '.join(ent))<=2 :
			found.append(poss)
	return found

