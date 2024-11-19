import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


# on study_type dataset:

# btw: 100 paper 4om full paper got study-type 0.52 (fast-sweep-1)

# beybert best tuned
kbt = [1,1,1,1,0,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,0]

# keybert new chunking (similar params to the rag one)
knc = [1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,0,1,1,0,1,0]

# keybert old chunking
koc = [1,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,1,1,0,1,1,0,0,0]

# rag old chunking
roc = [1,1,0,0,0,1,1,0,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1,0,1,0,0,0,0]

x1, x2 = roc, kbt

# on arxpr2 dataset (without sex_2)


# 100 papers, arxpr2_400, keybert/4om 4 chunks
a400 = [0,0,1,1,0,0,0,1,0,1,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,1.0]

# 100 papers, arxpr2_25, keybert/4om 4 chunks
a25= [0,1,0,0,0,1,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,1,1,0,1.0]

x1, x2 = a25,a400
print(len(x1))

a,b,c,d = 0,0,0,0
for i in range(len(x1)):
    a += x1[i] and x2[i]
    b += x1[i] and not x2[i]
    c += not x1[i] and x2[i]
    d += not x1[i] and not x2[i]

data = [[a, b], [c, d]]

print(data)


# all of them:
print(mcnemar(data, exact=True))
print(mcnemar(data, exact=False))
print(mcnemar(data, exact=False, correction=False))

# most correct(?)
print(mcnemar(data, exact=(b+c<25), correction=min(a,b,c,d)>25))


#x1 = np.array(x1)
#x2 = np.array(x2)
#
#print(ttest_rel(x1,x2))
#
#
#diff = x1-x2
#
print(np.mean(x1))
print(np.mean(x2))
#print(np.mean(diff))
#print(np.std(diff))
#print(diff)
