

import argparse
import glob
import os
import pickle
import json
from datetime import datetime

import numpy as np

from iqf_finetune import finetune_frcnn
from iqf_test import test_frcnn


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Perform IQF experiment on Faster R-CNN network')
  # parser.add_argument('--exp_name', dest='exp_name',
  #                     help='experiment name',
  #                     required=True, type=str)
  # TODO: support other datasets
  # parser.add_argument('--dataset', dest='dataset',
  #                     help='problem dataset',
  #                     default='sate_airports', type=str)
  # parser.add_argument('--modif', nargs='+', dest='modifiers',
  #                     help='image modifiers ie:JPG90 JPG80 ...',
  #                     default=['NULL'])
  # parser.add_argument('--n_runs', dest='n_runs',
  #                     help='number of IQF runs',
  #                     default=1, type=int)
  parser.add_argument('--n_epochs', dest='n_epochs',
                      help='number of epochs to train',
                      default=10, type=int)
  parser.add_argument('--min_plane_size', dest='min_plane_size',
                      help='min plane size to detect',
                      default=24, type=int)
  parser.add_argument('--trainds', dest='trainds',
                      help='path to training image files',
                      default='../datasets/SateAirports/PNGImages', required=True, type=str)
  parser.add_argument('--outputpath', dest='outputpath',
                      help='path to output folder',
                      default='./output/iqt', type=str)
  args = parser.parse_args()
  return args

def clean_files(del_pkl=False):
  os.system('rm data/cache/*')
  # remove previous ones except for the pre-trained model
  [os.remove('{}'.format(file)) for file in glob.glob('models/res101/sate_airports/*.pth') if 'faster_rcnn_1_7_10021.pth' not in file]
  os.system('rm -rf logs/*')
  os.system('rm data/SateAirports/ImageSets/Main/test.txt_annots.pkl')
  if del_pkl:
    os.system('rm output/res101/sate_airports_test/faster_rcnn_10/*')

def train_model(modif, n_epochs, ds_path=''):
  modif = '_' + modif if modif != 'NULL' else ''
  chkpnt = finetune_frcnn(args_dataset='sate_airports',
                 args_modif=modif,
                 args_net='res101',
                 args_max_epochs=n_epochs,
                 args_batch_size=4,
                 args_checksession=1,
                 args_checkepoch=7,
                 args_checkpoint=10021,
                 ds_path=ds_path)
  return chkpnt

def select_best_epoch(modif, n_epochs, checkpoint, val_set='test', min_plane_size=24, ds_path=''):
  modif = '_' + modif if modif != 'NULL' else ''
  aps = []
  for epoch in range(n_epochs):
    aps += [test_frcnn(args_dataset='sate_airports',
                      args_modif=modif,
                      args_net='res101',
                      args_checksession=1,
                      args_checkepoch=epoch + 1,
                      args_checkpoint=checkpoint,
                      args_vis=False,
                      output_results_files=False,
                      min_plane_size=min_plane_size,
                      ds_path=ds_path)[0]]
  return np.argmax(aps) + 1, np.amax(aps)

def test_model(run, modif, best_epoch, checkpoint, min_plane_size=24, ds_path=''):
  modif = '_' + modif if modif != 'NULL' else ''
  iqf_run = '_IQF' + str(run)
  test_frcnn(args_dataset='sate_airports',
                      args_modif=modif,
                      args_net='res101',
                      args_checksession=1,
                      args_checkepoch=best_epoch,
                      args_checkpoint=checkpoint,
                      args_vis=False,
                      output_results_files=True,
                      iqf_run=iqf_run,
                      min_plane_size=min_plane_size,
                      ds_path=ds_path)

def pack_results_pkl(args, exp_name=''):
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
                         'det': 'detections_img_id_' + run + '_' + mod + '.pkl'}
    run_dict['NULL'] = {'pr': 'aeroplane_pr_' + run + '.pkl',
                        'det': 'detections_img_id_' + run + '.pkl'}
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
  # with open(output_path + '/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + exp_name + '.pkl', 'wb') as f:
  #   pickle.dump(iqf, f, pickle.HIGHEST_PROTOCOL)

  
  detections = []
  for im in iqf[0]['NULL']['detections']:
    bboxes = []
    for d in im['boxes']:
      bboxes.append(list(d.astype(float)))
    detections.append({'img_id': im['img_id'], 'bboxes': bboxes})

  iqt_full_out_json = {'epochs': args.n_epochs,
                       'ds_path': args.trainds,
                       'min_plane_size': args.min_plane_size,
                       'rec': list(iqf[0]['NULL']['rec'].astype(float)),
                       'prec': list(iqf[0]['NULL']['prec'].astype(float)),
                       'ap': iqf[0]['NULL']['ap'].astype(float),
                       'detections': detections}
  results_json = {'min_plane_size': iqt_full_out_json['min_plane_size'],
                  'ap': iqt_full_out_json['ap']}

  output_json = []
  det_id = 1
  for img in iqt_full_out_json['detections']:
    for det in img['bboxes']:
        det_dict = {
            'image_id': int(img['img_id']),
            'iscrowd': 0,
            'bbox': [det[0],det[1],det[2]-det[0],det[3]-det[1]],
            'area': abs(det[0]-det[1])*abs(det[2]-det[3]),
            'category_id': 1,
            'id': det_id,
            'score': det[4]
        }
        output_json.append(det_dict)
        det_id +=1
  
  with open(args.outputpath + '/' + 'iqt_full_out.json', 'w') as f:
    json.dump(iqt_full_out_json, f)
  with open(args.outputpath + '/' + 'results.json', 'w') as f:
    json.dump(results_json, f)
  with open(args.outputpath + '/' + 'output.json', 'w') as f:
    json.dump(output_json, f)

def main(args):
  # if 'NULL' not in args.modifiers:
  #   args.modifiers.append('NULL')
  modifiers = ['NULL']
  print('Called with args:')
  print(args)
  clean_files(del_pkl=True)
  # for run in range(args.n_runs):
  for run in range(1):
    print('IQF experiment Num: ', run)
    # for modif in args.modifiers:
    for modif in modifiers:
      print('Modifier to test: ', modif)
      clean_files(del_pkl=False)
      checkpoint = train_model(modif, n_epochs=args.n_epochs, ds_path=args.trainds)
      best_epoch, best_ap = select_best_epoch(modif, args.n_epochs, checkpoint, val_set='test', min_plane_size=args.min_plane_size, ds_path=args.trainds)
      print('Best Epoch: ', best_epoch)
      print('Best AP: ', best_ap)
      test_model(run, modif, best_epoch, checkpoint, min_plane_size=args.min_plane_size, ds_path=args.trainds)
  # pack_results_pkl(exp_name=args.exp_name)
  pack_results_pkl(args, exp_name='iqt_experiment')
  clean_files(del_pkl=False)

if __name__ == '__main__':
  args = parse_args()
  main(args)
