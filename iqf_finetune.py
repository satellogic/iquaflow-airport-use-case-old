# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from termcolor import colored

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--modif', dest='modif',
                      help='image modifier ie: _JPG90',
                      default='', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of workers to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether to perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# load pretrained model
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and display
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

def finetune_frcnn(args_dataset='pascal_voc',
                   args_modif='',
                   args_net='vgg16',
                   args_start_epoch=1,
                   args_max_epochs=20,
                   args_disp_interval=100,
                   args_checkpoint_interval=10000,
                   args_save_dir="models",
                   args_num_workers=0,
                   args_cuda=True,
                   args_large_scale=False,
                   args_mGPUs=False,
                   args_batch_size=1,
                   args_class_agnostic=False,
                   args_optimizer="sgd",
                   args_lr=0.001,
                   args_lr_decay_step=5,
                   args_lr_decay_gamma=0.1,
                   args_session=1,
                   args_load_dir="models",
                   args_resume=False,
                   args_checksession=1,
                   args_checkepoch=1,
                   args_checkpoint=0,
                   args_use_tfboard=False,
                   ds_path=''):

  print('Finetune FRCNN...')

  if args_dataset == "pascal_voc":
      args_imdb_name = "voc_2007_trainval"
      args_imdbval_name = "voc_2007_test"
      args_set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args_dataset == "pascal_voc_0712":
      args_imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args_imdbval_name = "voc_2007_test"
      args_set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args_dataset == "coco":
      args_imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args_imdbval_name = "coco_2014_minival"
      args_set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args_dataset == "imagenet":
      args_imdb_name = "imagenet_train"
      args_imdbval_name = "imagenet_val"
      args_set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args_dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args_imdb_name = "vg_150-50-50_minitrain"
      args_imdbval_name = "vg_150-50-50_minival"
      args_set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args_dataset == "sate_airports":
      args_imdb_name = "sate_airports_trainval"
      args_imdbval_name = "sate_airports_test"
      # TODO: Entender que son estos parametros
      args_set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '40']

  args_cfg_file = "cfgs/{}_ls.yml".format(args_net) if args_large_scale else "cfgs/{}.yml".format(args_net)

  if args_cfg_file is not None:
    cfg_from_file(args_cfg_file)
  if args_set_cfgs is not None:
    cfg_from_list(args_set_cfgs)

  # print('Using config:')
  # pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args_cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args_cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args_imdb_name, args_modif, ds_path=ds_path)
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))

  # ARQUELOGIA DE DATALOADING
  # print('type(imdb): ', type(imdb))
  # print('type(imdb.num_classes): ', type(imdb.num_classes))
  # print('imdb.num_classes: ', imdb.num_classes)
  # print('type(imdb.classes): ', type(imdb.classes))
  # print('imdb.classes: ', imdb.classes)
  # print('type(roidb): ', type(roidb))
  # print('type(roidb[0]): ', type(roidb[0]))
  # print('Keys of roidb[0]:')
  # print(roidb[0].keys())
  # print('Contents of roidb[0]:')
  # print(roidb[0])
  # print('type(ratio_list): ', type(ratio_list))
  # print('ratio_list.shape: ', ratio_list.shape)
  # print('type(ratio_index): ', type(ratio_index))
  # print('ratio_index.shape: ', ratio_index.shape)

  # assert False

  #


  output_dir = args_save_dir + "/" + args_net + "/" + args_dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args_batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args_batch_size, \
                           imdb.num_classes, training=True)

  
  # ARQUELOGIA DE DATALOADING
  # print('type(dataset): ', type(dataset))

  # assert False
  #
  
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_batch_size,
                            sampler=sampler_batch, num_workers=args_num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args_cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args_cuda:
    cfg.CUDA = True


  pascal_classes = np.asarray(['__background__',
                               'aeroplane', 'bicycle', 'bird', 'boat',
                               'bottle', 'bus', 'car', 'cat', 'chair',
                               'cow', 'diningtable', 'dog', 'horse',
                               'motorbike', 'person', 'pottedplant',
                               'sheep', 'sofa', 'train', 'tvmonitor'])

  # initilize the network here.
  if args_net == 'vgg16':
    # fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args_class_agnostic)
    fasterRCNN = vgg16(pascal_classes, pretrained=True, class_agnostic=args_class_agnostic)
  elif args_net == 'res101':
    # fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args_class_agnostic)
    fasterRCNN = resnet(pascal_classes, 101, pretrained=True, class_agnostic=args_class_agnostic)
  elif args_net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args_class_agnostic)
  elif args_net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args_class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()



  fasterRCNN.create_architecture()


  # FINE TUNING
  # Load pretrained model checkpoint
  load_name = os.path.join(output_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args_checksession, args_checkepoch, args_checkpoint))
  print("loading checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  # args_session = checkpoint['session']
  # args_start_epoch = checkpoint['epoch']
  fasterRCNN.load_state_dict(checkpoint['model'])
  # optimizer.load_state_dict(checkpoint['optimizer'])
  # lr = optimizer.param_groups[0]['lr']
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  print("loaded checkpoint %s" % (load_name))


  # print('RED ORIGINAL:')
  # print(fasterRCNN)
  print('ULTIMAS DOS CAPAS ORIGINALES:')
  print(fasterRCNN.RCNN_cls_score)
  print(fasterRCNN.RCNN_bbox_pred)

  ###########################################
  # FREEZAR LAYERS
  #   FINE TUNING --> comentar
  #   FEATURE EXTRACTION --> descomentar
  #
  # for param in fasterRCNN.parameters():
  #           param.requires_grad = False
  ###########################################

  RCNN_in_features = fasterRCNN.RCNN_cls_score.in_features

  fasterRCNN.RCNN_cls_score = nn.Linear(RCNN_in_features, imdb.num_classes)
  fasterRCNN.RCNN_bbox_pred = nn.Linear(RCNN_in_features, 4 * imdb.num_classes)
  # print('RED MODIFICADA:')
  # print(fasterRCNN)
  print('ULTIMAS DOS CAPAS MODIFICADAS:')
  print(fasterRCNN.RCNN_cls_score)
  print(fasterRCNN.RCNN_bbox_pred)

  
  print('PARAMETROS A TUNEAR:')
  # for name,param in fasterRCNN.named_parameters():
  #       if param.requires_grad == True:
  #           print("\t",name)
  print(len([_ for _, param in fasterRCNN.named_parameters() if param.requires_grad == True]))

  # assert False
  #


  lr = cfg.TRAIN.LEARNING_RATE
  lr = args_lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args_momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args_optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args_optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args_cuda:
    fasterRCNN.cuda()

  if args_resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args_checksession, args_checkepoch, args_checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args_session = checkpoint['session']
    args_start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args_mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args_batch_size)

  if args_use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args_start_epoch, args_max_epochs + 1):
    # setting to train mode
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args_lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args_lr_decay_gamma)
        lr *= args_lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      
      # ARQUELOGIA DE DATALOADING
      # print('type(data): ', type(data))
      # print('len(data): ', len(data))
      # print('type(data[0]): ', type(data[0]))
      # print('type(data[1]): ', type(data[1]))
      # print('type(data[2]): ', type(data[2]))
      # print('type(data[3]): ', type(data[3]))

      # print('Size of data[0]:')
      # print(data[0].size())
      # print('Content of data[0]:')
      # print(data[0])

      # print('Size of data[1]:')
      # print(data[1].size())
      # print('Content of data[1]:')
      # print(data[1])

      # print('Size of data[2]:')
      # print(data[2].size())
      # print('Content of data[2]:')
      # print(data[2])

      # print('Size of data[3]:')
      # print(data[3].size())
      # print('Content of data[3]:')
      # print(data[3])

      # assert False
      #
      
      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args_net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args_disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args_disp_interval + 1)

        if args_mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] %s, lr: %.2e" \
                                % (args_session, epoch, step, iters_per_epoch, colored('loss: %.4f' % loss_temp, 'green', attrs=['bold']), lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args_use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args_session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args_session, epoch, step))
    save_checkpoint({
      'session': args_session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args_mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args_class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args_use_tfboard:
    logger.close()

  return step 
