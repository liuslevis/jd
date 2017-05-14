import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

action_path = 'data/raw/JData_Action_201603.csv'

a = np.zeros((7,31), dtype=np.int64)

with open(action_path) as f:
    for line in f.readlines():
        if 'user_id,sku_id' in line: continue
        # user_id,sku_id,time,model_id,type,cate,brand
        parts = line.split(',') 
        dt = int(parts[2].split(' ')[0].replace('-', '')) - 20160300
        act = int(parts[4])
        if 1 <= dt <= 30 and 1<= act <= 7:
            a[act][dt] += 1

fig = plt.figure()

for i in range(1,7):
    ax = fig.add_subplot(710 + i)
    ax.set_title('act_%d' % i)
    ax.set_autoscale_on(True)
    ax.plot(a[i])


ctr = a[4] / a[1]
ax = fig.add_subplot(710 + 7)
ax.set_title('act_4 / act_1' )
ax.set_autoscale_on(True)
ax.plot(ctr)

plt.show()

# plt.plot(a[4])
# plt.plot(a[1])
# plt.plot(pv)
# plt.show()

