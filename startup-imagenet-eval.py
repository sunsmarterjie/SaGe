import ast
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=None, help='the output path')
# ./configs/benchmarks/linear_classification/imagenet/r50_last_byol.py
parser.add_argument('--config', type=str,
                    default='./configs/benchmarks/linear_classification/imagenet/r50_last_byol.py',
                    help='the path of the config file')
parser.add_argument('--checkpoint', type=str, default='/your_path/epoch_300.pth', help='the path of the ckp')
parser.add_argument('--convert', type=ast.literal_eval, default='True',
                    help='wheter convert the checkpoint file into the detectron2 format')
args, unparsed = parser.parse_known_args()

print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))
os.chdir('./SaGe_code')
print('Current path changed to: ' + os.getcwd())

print('Start pip install')
os.system('pip install -q -e .')
print('Finish pip install')


print('Start converting ckp!')
if args.convert:
    print('convert')
    os.system('python tools/extract_backbone_weights.py %s backbone.pth.tar' % args.checkpoint)
    ckp_file = 'backbone.pth.tar'
print('Finish converting ckp!')

os.system('ln -s ../data_path/imagenet/train ./data/imagenet/train')
os.system('ln -s ../data_path/imagenet/val ./data/imagenet/val')

os.system(
    'bash benchmarks/dist_train_linear_eval.sh %s %s %s --use_mox False' % (args.config, ckp_file, args.train_url))
