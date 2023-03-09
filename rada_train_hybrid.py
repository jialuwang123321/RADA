import argparse
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
import os
import os.path as osp

import dsacstar
from dataset import CamLocDataset
from dsacstar import Network

from AT.rada_attack import RADA_Attack
from AT.rada_pytorch import PyTorchClassifier

parser = argparse.ArgumentParser(
	description='Train scene coordinate regression in an end-to-end fashion.',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')

parser.add_argument('network_in', help='file name of a network initialized for the scene')

#parser.add_argument('network_out', help='output file name for the new network')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
	help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
	help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
	help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--learningrate', '-lr', type=float, default=0.000001, 
	help='learning rate')

parser.add_argument('--iterations', '-it', type=int, default=100000, 
	help='number of training iterations, i.e. network parameter updates')

parser.add_argument('--weightrot', '-wr', type=float, default=1.0, 
	help='weight of rotation part of pose loss')

parser.add_argument('--weighttrans', '-wt', type=float, default=100.0, 
	help='weight of translation part of pose loss')

parser.add_argument('--softclamp', '-sc', type=float, default=100, 
	help='robust square root loss after this threshold')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, 
	help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[1,2],
	help='test mode: 1 = RGB, 2 = RGB-D')

parser.add_argument('--tiny', '-tiny', action='store_true',
	help='Train a model with massively reduced capacity for a low memory footprint.')

parser.add_argument('--session', '-sid', default='',
	help='custom session name appended to output files. Useful to separate different runs of the program')

parser.add_argument('--weightrot_rada', '-wr_rada', type=float, default=1.0, 
	help='weight of rotation part of pose loss_rada')

parser.add_argument('--weighttrans_rada', '-wt_rada', type=float, default=100.0, 
	help='weight of translation part of pose loss_rada')

parser.add_argument('--softclamp_rada', '-sc_rada', type=float, default=100, 
	help='robust square root loss after this threshold_rada')

parser.add_argument('--threshold_rada', type=int, default=2, 
	help='The number of threshold for RADA perturbation, see Paper Section III-B Step 4')

parser.add_argument('--eps_rada', type=float, default=158.0, 
	help='The scaling factors for RADA perturbation, see Paper Section III-B Step 3')

parser.add_argument('--pow_rada', type=float, default=1.5, 
	help='The value of pow for RADA perturbation, see Paper Section III-B Step 3')

opt = parser.parse_args()

trainset = CamLocDataset("./datasets/" + opt.scene + "/train", mode=(0 if opt.mode < 2 else opt.mode), augment=True, aug_rotation=0, aug_scale_min=1, aug_scale_max=1) # use only photometric augmentation, not rotation and scaling
trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=16)

print("Found %d training images for %s." % (len(trainset), opt.scene))

# load network
network = Network(torch.zeros((3)), opt.tiny)
network.load_state_dict(torch.load(opt.network_in))
network = network.cuda()
network.train()

print("Successfully loaded %s." % opt.network_in)

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

# RADA_Step_1: define rada_dsac loss
class radadsace2eLoss(nn.Module):
    def __init__(self):
        super(radadsace2eLoss, self).__init__()

    def forward(self, pred_coordinates):
        loss = pred_coordinates
        return loss

radadsace2eloss = radadsace2eLoss()

# RADA_Step_2: Create the ART classifier 
classifier = PyTorchClassifier(
    model=network,  
    loss=radadsace2eloss, 
    optimizer=optimizer,
    input_shape=(1,1,480,640), 
    nb_classes=1, 
)

# RADA_Step_3: Define RADA attack
attack = RADA_Attack(estimator=classifier, eps=opt.eps_rada, batch_size=1, optuna_pow = opt.pow_rada,perturb_threshold = opt.threshold_rada)

iteration = 0
epochs = int(opt.iterations / len(trainset))

# keep track of training progress
train_log = open('log_e2e_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)

training_start = time.time()

for epoch in range(epochs):	

	print("=== Epoch: %7d ======================================" % epoch)

	for image, pose, camera_coordinates, focal_length, file in trainset_loader:
            
		start_time = time.time()
		focal_length = float(focal_length[0])
		pose = pose[0]

		if (epoch % 2) != 0:  #odd number of epoch: Train with RADA Perturbation
			# Stage 1 with origional images: predict scene coordinates
			scene_coordinates = network(image.cuda())
			scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

			# Stage 2 with origional images: RANSAC, return loss and grad
			if opt.mode == 2:
				# RGB-D mode
				loss = dsacstar.backward_rgbd(
					scene_coordinates.cpu(), 
					camera_coordinates,
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					opt.weightrot_rada,
					opt.weighttrans_rada,
					opt.softclamp_rada,
					opt.inlieralpha,
					opt.maxpixelerror,
					random.randint(0,1000000)) # used to initialize random number generator in C++

			else:
				# RGB mode
				loss = dsacstar.backward_rgb(
					scene_coordinates.cpu(), 
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					focal_length, 
					float(image.size(3) / 2), #principal point assumed in image center
					float(image.size(2) / 2),
					opt.weightrot_rada,
					opt.weighttrans_rada,
					opt.softclamp_rada,
					opt.inlieralpha,
					opt.maxpixelerror,
					network.OUTPUT_SUBSAMPLE,
					random.randint(0,1000000)) # used to initialize random number generator in C++


			# RADA_Step_4: Generate RADA perturbed Images
			image = attack.generate_ransac(image.numpy(), pose.numpy(), scene_coordinate_gradients, perturb_threshold = opt.threshold_rada ,optuna_eps = opt.eps_rada ,optuna_pow = opt.pow_rada)
			image = torch.from_numpy(image)

			# RADA_Step_5: predict scene coordinates
			scene_coordinates = network(image.cuda())
			scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

			# RADA_Step_6: RANSAC, return loss and grad
			if opt.mode == 2:
				# RGB-D mode
				loss = dsacstar.backward_rgbd(
					scene_coordinates.cpu(), 
					camera_coordinates,
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					opt.weightrot,
					opt.weighttrans,
					opt.softclamp,
					opt.inlieralpha,
					opt.maxpixelerror,
					random.randint(0,1000000)) # used to initialize random number generator in C++

			else:
				# RGB mode
				loss = dsacstar.backward_rgb(
					scene_coordinates.cpu(), 
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					focal_length, 
					float(image.size(3) / 2), #principal point assumed in image center
					float(image.size(2) / 2),
					opt.weightrot,
					opt.weighttrans,
					opt.softclamp,
					opt.inlieralpha,
					opt.maxpixelerror,
					network.OUTPUT_SUBSAMPLE,
					random.randint(0,1000000)) # used to initialize random number generator in C++

		else:
			# Stage 1 with origional images: predict scene coordinates
			scene_coordinates = network(image.cuda())
			scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

			# Stage 2 with origional images: RANSAC, return loss and grad
			if opt.mode == 2:
				# RGB-D mode
				loss = dsacstar.backward_rgbd(
					scene_coordinates.cpu(), 
					camera_coordinates,
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					opt.weightrot,
					opt.weighttrans,
					opt.softclamp,
					opt.inlieralpha,
					opt.maxpixelerror,
					random.randint(0,1000000)) # used to initialize random number generator in C++

			else:
				# RGB mode
				loss = dsacstar.backward_rgb(
					scene_coordinates.cpu(), 
					scene_coordinate_gradients,
					pose, 
					opt.hypotheses, 
					opt.threshold,
					focal_length, 
					float(image.size(3) / 2), #principal point assumed in image center
					float(image.size(2) / 2),
					opt.weightrot,
					opt.weighttrans,
					opt.softclamp,
					opt.inlieralpha,
					opt.maxpixelerror,
					network.OUTPUT_SUBSAMPLE,
					random.randint(0,1000000)) # used to initialize random number generator in C++

		# Update network parameters
		torch.autograd.backward((scene_coordinates), (scene_coordinate_gradients.cuda()))
		optimizer.step()
		optimizer.zero_grad()
		
		end_time = time.time()-start_time
		#print('Iteration: %6d, Loss: %.2f, Time: %.2fs \n' % (iteration, loss, end_time), flush=True)

		train_log.write('%d %f\n' % (iteration, loss))
		iteration = iteration + 1
	
	# Save model 
	if opt.mode == 2:
		filename = osp.join(os.getcwd(), 'models', opt.scene, 'e2e/rgbd','{}_rada_ransac_hybrid_epoch_{:03d}.net'.format(opt.scene, epoch))
		print('Saving snapshot of the network to %s.' % filename)
		torch.save(network.state_dict(), filename)
	else:
		filename = osp.join(os.getcwd(), 'models', opt.scene, 'e2e/rgb','{}_rada_ransac_hybrid_epoch_{:03d}.net'.format(opt.scene, epoch))
		print('Saving snapshot of the network to %s.' % filename)
		torch.save(network.state_dict(), filename)


print('Done without errors. Time: %.1f minutes.' % ((time.time() - training_start) / 60))
train_log.close()








