import argparse
import glob
import os
import pickle
from datetime import datetime

import numpy as np

from iqf_finetune import finetune_frcnn
from iqf_test import test_frcnn


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Perform IQF experiment on Faster R-CNN network')
  parser.add_argument('--exp_name', dest='exp_name',
                      help='experiment name',
                      required=True, type=str)
  # TODO: support other datasets
  # parser.add_argument('--dataset', dest='dataset',
  #                     help='problem dataset',
  #                     default='sate_airports', type=str)
  parser.add_argument('--modif', nargs='+', dest='modifiers',
                      help='image modifiers ie:JPG90 JPG80 ...',
                      default=['NULL'])
  parser.add_argument('--n_runs', dest='n_runs',
                      help='number of IQF runs',
                      default=1, type=int)
  parser.add_argument('--n_epochs', dest='n_epochs',
                      help='number of epochs to train',
                      default=10, type=int)
  parser.add_argument('--min_plane_size', dest='min_plane_size',
                      help='min plane size to detect',
                      default=24, type=int)
  args = parser.parse_args()
  return args

def clean_files(del_pkl=False):
  # os.system('rm test_imgs/*_det*')
  os.system('rm data/cache/*')
  os.system('rm models/res101/sate_airports/*_40*')
  os.system('rm -rf logs/*')
  os.system('rm data/SateAirports/ImageSets/Main/test.txt_annots.pkl')
  if del_pkl:
    os.system('rm output/res101/sate_airports_test/faster_rcnn_10/*')

def train_model(modif, n_epochs):
  modif = '_' + modif if modif != 'NULL' else ''
  finetune_frcnn(args_dataset='sate_airports',
                 args_modif=modif,
                 args_net='res101',
                 args_max_epochs=n_epochs,
                 args_batch_size=4,
                 args_checksession=1,
                 args_checkepoch=7,
                 args_checkpoint=10021)

def select_best_epoch(modif, n_epochs, val_set='test', min_plane_size=24):
  modif = '_' + modif if modif != 'NULL' else ''
  aps = []
  for epoch in range(n_epochs):
    aps += [test_frcnn(args_dataset='sate_airports',
                      args_modif=modif,
                      args_net='res101',
                      args_checksession=1,
                      args_checkepoch=epoch + 1,
                      args_checkpoint=40,
                      args_vis=False,
                      output_results_files=False,
                      min_plane_size=min_plane_size)[0]]
  return np.argmax(aps) + 1, np.amax(aps)

def test_model(run, modif, best_epoch, min_plane_size=24):
  modif = '_' + modif if modif != 'NULL' else ''
  iqf_run = '_IQF' + str(run)
  test_frcnn(args_dataset='sate_airports',
                      args_modif=modif,
                      args_net='res101',
                      args_checksession=1,
                      args_checkepoch=best_epoch,
                      args_checkpoint=40,
                      args_vis=False,
                      output_results_files=True,
                      iqf_run=iqf_run,
                      min_plane_size=min_plane_size)

def pack_results_pkl(output_path='output/iqf/', exp_name=''):
  input_path = 'output/res101/sate_airports_test/faster_rcnn_10/'
  file_names = [name.split('/')[-1].split('.')[0] for name in glob.glob(input_path + "*")]
  runs = set(expr for name in file_names for expr in name.split('_') if 'IQF' in expr)
  mods = set(name.split('_')[-1] for name in file_names)
  mods = [mod for mod in mods if 'IQF' not in mod]
  iqf = []
  for run in runs:
    run_dict = {}
    for mod in mods: 
        run_dict[mod] = {'pr': 'aeroplane_pr_' + run + '_' + mod + '.pkl',
                         'det': 'detections_' + run + '_' + mod + '.pkl'}
    run_dict['NULL'] = {'pr': 'aeroplane_pr_' + run + '.pkl',
                        'det': 'detections_' + run + '.pkl'}
    iqf.append(run_dict)
  for run in iqf:
    for mod in run:
        with open(input_path + run[mod]['pr'], 'rb') as f:
            pkl_file = pickle.load(f)
            run[mod].update(pkl_file)
        with open(input_path + run[mod]['det'], 'rb') as f:
            pkl_file = pickle.load(f)
            run[mod]['detections'] = pkl_file[1]
        del(run[mod]['pr'])
        del(run[mod]['det'])
  with open(output_path + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + exp_name + '.pkl', 'wb') as f:
    pickle.dump(iqf, f, pickle.HIGHEST_PROTOCOL)

def main(args):
  if 'NULL' not in args.modifiers:
    args.modifiers.append('NULL')
  print('Called with args:')
  print(args)
  clean_files(del_pkl=True)
  for run in range(args.n_runs):
    print('IQF experiment Num: ', run)
    for modif in args.modifiers:
      print('Modifier to test: ', modif)
      clean_files(del_pkl=False)
      train_model(modif, n_epochs=args.n_epochs)
      best_epoch, best_ap = select_best_epoch(modif, n_epochs=args.n_epochs, val_set='test', min_plane_size=args.min_plane_size)
      print('Best Epoch: ', best_epoch)
      print('Best AP: ', best_ap)
      test_model(run, modif, best_epoch, min_plane_size=args.min_plane_size)
  pack_results_pkl(exp_name=args.exp_name)
  clean_files(del_pkl=False)

if __name__ == '__main__':
  args = parse_args()
  main(args)
