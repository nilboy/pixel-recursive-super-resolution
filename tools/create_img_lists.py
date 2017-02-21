"""Create image-list file
Example:
python tools/create_img_lists.py --dataset=data/celebA --outfile=data/train.txt
"""
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--dataset", dest="dataset",  
                  help="dataset path")

parser.add_option("--outfile", dest="outfile",  
                  help="outfile path")
(options, args) = parser.parse_args()

f = open(options.outfile, 'w')
dataset_basepath = options.dataset
for p1 in os.listdir(dataset_basepath):
  image = os.path.abspath(dataset_basepath + '/' + p1)
  f.write(image + '\n')
f.close()
