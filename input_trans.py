'''
Transform the type of training data in "TrainDataset" to the type of groundtruth in "ICDAD2015"
'''
import os
import json

json_path = os.listdir('./TrainDataset/json')
for file in json_path:
  tmp = './TrainDataset/json' + '/' + file
  with open(tmp, newline='') as jsonfile:
    ori = json.load(jsonfile)
    
    tmp = './TrainDataset/gt/gt_' + file[:-5] + '.txt'
    new = open(tmp, 'w')
    for i in ori['shapes']:
      for j in i['points']:
        for p in j:
          new.write(str(int(p)))
          new.write(',')
      new.write('#')
      new.write('\n')
    new.close()
