'''
Tansform the type of model pred in 'PublicTestDataset/pred' to the type wanted by the competition
'''
import os
import pandas as pd

pred = os.listdir('./PublicTestDataset/pred')

def orderchange(x):
    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] = x[6], x[7], x[4], x[5], x[2], x[3], x[0], x[1]
    return x

numbers = []
coordinates = []
confidence = []
for filename in pred:
    if '.txt' in filename:
        f = open(os.path.join('./PublicTestDataset/pred/', filename))
        for line in f:
            co = line[:-1].split(',') # 8 coordinates
            co = orderchange(co) # list type
            co = ','.join(co) # str type
            
            numbers.append(int(filename[8:-4]))
            coordinates.append(co)
            confidence.append(1.0)

zipped_list = zip(numbers, coordinates, confidence)
sorted_pair = sorted(zipped_list)
tuples = zip(*sorted_pair)
numbers, coordinates, confidence = [list(tuple) for tuple in tuples]

dic = {'frameNumber':numbers, 'coordinates':coordinates, 'confidence':confidence}
output = pd.DataFrame(dic)
output.to_csv('./PublicTestDataset/pred/output.csv', encoding='utf-8', index=False)
