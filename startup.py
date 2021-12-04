import os
import moxing as mox
import argparse
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default='', help='the output path')
parser.add_argument('--config', type=str, default=None, help='the path of the config file')
parser.add_argument('--out_path', type=str, default=None, help='the path of the config file')
args, unparsed = parser.parse_known_args()

print('train_url:', args.train_url)
# ############# preparation stage ####################
print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))
print()
os.chdir('./SaGe_code')
print('Current path changed to: ' + os.getcwd())

print('Start pip install')
os.system('pip install -q -e .')
print('Finish pip install')


os.system('ln -s ../data_path/imagenet/train ./data/imagenet/train')

# ############# preparation stage ####################

os.system('sh ./tools/dist_train.sh %s 8 ' % (args.config) + str(args.out_path) + ' --use_mox True')
