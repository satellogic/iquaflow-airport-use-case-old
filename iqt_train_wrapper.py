
import argparse
import os

cuda_vis_dev = "0,1"
python_path = "/miniconda3/envs/airport-env/bin/python"
n_epochs = 4
min_plane_size = 11
outputpath = "/Data/share/out-iq-sateairports/test1"
trainds = "/Data/share/SateAirports/"

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	# These are modified by IQF
	#"python {} --trainds {} --valds {} --outputpath {} {}"
	#"python {} --trainds {} --outputpath {} {}"
	parser.add_argument('--trainds', type=str, default=trainds, help='input dataset path')
	parser.add_argument('--outputpath', type=str, default=outputpath, help='input dataset path')
	
	# These are fixed for now
	parser.add_argument('--cu', type=str, default=cuda_vis_dev, help='CUDA_VISIBLE_DEVICES')
	parser.add_argument('--py', type=str, default=python_path, help='Full python path')
	parser.add_argument('--nep', type=int, default=n_epochs, help='Number of epochs')
	parser.add_argument('--min_size', type=int, default=min_plane_size, help='Minimum plane size')
	
	opt = parser.parse_args()
	
	cuda_vis_dev = opt.cu
	trainds = os.path.join(opt.trainds,'PNGImages')
	outputpath = opt.outputpath
	n_epochs = opt.nep
	min_size = opt.min_size
	
	cmd = ""
	cmd+=f"CUDA_VISIBLE_DEVICES={cuda_vis_dev} && {python_path} iqt_train.py "
	cmd+=f"--trainds {trainds} "
	cmd+=f"--outputpath {outputpath} "
	cmd+=f"--n_epochs={n_epochs} --min_plane_size={min_size}"
	
	with open('./wrapper.log','w') as f:
		f.write(cmd)
	os.system(cmd)
