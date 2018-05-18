import numpy as np


# concat all the data in atemp
i=1
filename='extract/train-{}.npz'.format(i)
afeattemp=np.load(filename)['testdata']
afeattemp=afeattemp.reshape(-1,1)
atemp = afeattemp

for i in range(2,10):
    filename='extract/train-{}.npz'.format(i)
    afeattemp=np.load(filename)['testdata']
    afeattemp=afeattemp.reshape(-1,1)
    atemp=np.vstack((atemp,afeattemp))
    # np.random.shuffle(atemp)
    print("finish {}".format(i))

# get the number of samples in each class
b = np.zeros(11)
for i in range(len(atemp)):
    b[atemp[i][0]['targets']] += 1

#classify the dataset
a = []
for i in range(11):
    a.append(np.zeros(0,1))

for i in range(len(atemp)):
    target = atemp[i][0]['targets']
    a[target] = np.vstack((a[target], atemp[i]))

del atemp

# sample
datatotal  = np.zeros((0,1))
for i in range(11):
    for j in range(10000):
        randindex = np.random.randind(0,len(a[i]))
        datatotal = np.vstack((datatotal, a[i][randindex]))
# shuffle
np.random.shuffle(datatotal)

del a

subset = 40000
temp1 = datatotal[0:40000]
np.savez('trainset1.npz',data= temp1)
temp2 = datatotal[40000:80000]
np.savez('trainset2.npz',data= temp2)
temp3 = datatotal[80000:]
np.savez('trainset3.npz',data= temp3)

train_audio = [datatotal[i][0]['lmfcc'] for i in range(datatotal.shape[0])]
np.mean(train_audio)
mean_of_datatotal = -2.643583579852149