import ast
import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--train_url', type=str, default=' ', help='the output path')
parser.add_argument('--out_path', type=str, default=None, help='the path of the config file')
parser.add_argument('--config', type=str, default=None, help='the path of the config file')

parser.add_argument('--num_gpus', type=int, default=8, help='the number of gpus')
parser.add_argument('--rank', type=int, default=0, help='node rank')
parser.add_argument('--world_size', type=int, default=4, help='world size')

args, unparsed = parser.parse_known_args()

# ############# preparation stage ####################
print('Current path: ' + os.getcwd())
print('Current dirs: ' + str(list(os.listdir())))
print()
os.chdir('./SaGe_code')
print('Current path changed to: ' + os.getcwd())

print('Start pip install')
os.system('pip install -q -e .')
print('Finish pip install')

os.system('pip install --ignore-installed PyYAML')
os.system(
    'pip install -q --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex/')

master_host = os.environ['VC_WORKER_HOSTS'].split(',')[0]
master_addr = master_host.split(':')[0]
master_port = '8524'
# FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
# FLAGS.rank will be re-computed in main_worker
rank = args.rank  # ModelArts receive FLAGS.rank means node_rank
world_size = args.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
os.environ['MASTER_ADDR'] = master_addr
os.environ['MASTER_PORT'] = master_port

os.system('ln -s ../data_path/imagenet/train ./data/imagenet/train')

cmd_str = f"python -m torch.distributed.launch --nproc_per_node {args.num_gpus} \
    --nnodes={world_size} --node_rank={rank} --master_addr={master_addr} \
    --master_port={master_port} ./tools/train.py --config {args.config} --work_dir {args.out_path} \
    --seed 0 --launcher pytorch --use_mox True"

print('The running command is: ' + cmd_str)

os.system(cmd_str)
