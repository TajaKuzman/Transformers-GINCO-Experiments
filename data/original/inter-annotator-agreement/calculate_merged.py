import krippendorff
import numpy as np
from collections import Counter
file=open('IAA_sheet.txt')
file.readline()
labels=[[],[]]
for line in file:
	eid,a1c1,a1c2,a2c1,a2c2,aggr=line.split('\t')
	labels[0].append('|'.join(sorted((a1c1,a1c2))))
	labels[1].append('|'.join(sorted((a2c1,a2c2))))
distr=dict([(b,a) for a,b in enumerate(list(Counter(labels[0]+labels[1]).keys()))])
labels[0]=[distr[e] for e in labels[0]]
labels[1]=[distr[e] for e in labels[1]]
print(krippendorff.alpha(reliability_data=labels,level_of_measurement='nominal'))
