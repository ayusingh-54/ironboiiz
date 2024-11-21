from os.path import join, basename
from options.ironboiiz.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
from data.transforms import __scale_width
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util


opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log =True
opt.display_id=0
opt.verbose = False

datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'



eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_sir2 = datasets.CEILTestDataset(join(datadir, 'sir2_withgt'))

eval_dataset_real = datasets.CEILTestDataset(
    join(datadir, 'real20'),
    fns=read_fns('real_test.txt'),
    size=20)

eval_dataset_postcard = datasets.CEILTestDataset(join(datadir, 'postcard'))
eval_dataset_solidobject = datasets.CEILTestDataset(join(datadir, 'solidobject'))




eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_sir2 = datasets.DataLoader(
    eval_dataset_sir2, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard, batch_size=1, shuffle=False,
    num_workers=opt.nThreads, pin_memory=True)




engine = Engine(opt)

"""Main Loop"""
result_dir = './results'


res = engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2', savedir=join(result_dir, 'CEILNet_table2'))

