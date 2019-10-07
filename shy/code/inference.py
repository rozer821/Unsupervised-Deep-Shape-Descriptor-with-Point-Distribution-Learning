import argparse,os,sys,torch
import numpy as np
sys.path.append('./model/')
from model import *
sys.path.append('./utils/')
from functions import *


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers [default: 8]')
parser.add_argument('--model_path', default="./checkpoint",required=True, help='model checkpoint file path')
parser.add_argument('--dump_dir', default="./save",required=True, help='dump folder path')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
opt = parser.parse_args()

BATCH_SIZE = opt.batch_size
NUM_POINT = opt.num_point
MODEL_PATH = opt.model_path
GPU_INDEX = opt.gpu
DUMP_DIR = opt.dump_dir
WORKERS=opt.workers
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(opt) + '\n')
NUM_CLASSES = 13
MODEL_FILE = os.path.join('./models', opt.model+'.py')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    dataset = VOC()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=WORKERS)
    network = MF_GeoNet()
    network.cuda()
    network.load_state_dict(torch.load(opt.model))
    network.eval()

    avgpck=[]
    for i, data in enumerate(dataloader, 0):
        source, target, gt = data
        points, target,gt = Variable(points), Variable(target),Variable(gt)
        points,target = points.transpose(2, 1),target.transpose(2,1)
        points, target = points.cuda(), target.cuda()
        offset= network(points)
        trans=image_wapper(source,offset)
        pck=gtPCK(trans,gt)
        avgpck.append(pck)
        log_string('i:%d  loss: %f pck: %f' % (i, loss.data.item(), pck))
    avgpck=np.array(avgpck)
    log_string('Avg.PCK: %d' %(np.mean(avgpck)))

if __name__ == '__main__':
    evaluate()