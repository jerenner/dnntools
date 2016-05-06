import logging

# ---------------------------------------------------------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------------------------------------------------------

# Directory and run names
datdir = "/home/jrenner/data"                          # the base data directory
dname = "vox_dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200" # the data name
#dname = "vox_dnn3d_MAGBOX_noBrem_B05_21k_v2x2x2_r200x200x200"
rdir = "/home/jrenner/dnn/run"                             # the run directory
#rname = "dnn_10x10x10_zslice_test"     # the run name
rname = "dnn_GEANT_10x10x10_3d"

# Net configuration parameters
net_name = "MNISTadv3d"                    # name of the neural net described in nets/neuralnets.py
read_googlenet = False                        # set to true only if using the GoogLeNet
train_init = True                             # if true, train from net with standard pre-training; if false, read in a previously trained net

# Voxel parameters
vox_ext = 200          # extent of voxelized region in 1 dimension (in mm)
vox_sizeX = 10         # voxel size x (in mm)
vox_sizeY = 10         # voxel size y (in mm)
vox_sizeZ = 10         # voxel size z (in mm)
vox_norm = 1.0         # voxel normalization
nchannels = 1         # number of channels (will be automatically set to 3 if using the projections); if < 0, use 3D
use_proj = False
use_3d = True
vox_size = 10          # voxel size (for old scripts)

# Parameters describing training intervals and number of events for training and validation
ntrain_evts = 4600    # number of training evts per dataset
nval_evts = 200       # number of validation events
num_epochs = 20        # total number of epochs to train
epoch_blk_size = 1     # number of epochs to run per block (before reading new dataset); set equal to num_epochs unless data to be read in multiple blocks
dtblk_size = 4600      # number of signal and background events per training block
batch_size = 200       # training batch size

# Training optimizer parameters
opt_lr = 1.0e-3        # optimizer learning rate
opt_eps = 1.0e-6       # optimizer epsilon (for AdamOptimizer)
opt_mom = 0.9          # optimizer momentum
opt_decaybase = 0.2    # multiplicative factor for learning rate decay
opt_ndecayepochs = 40  # decay interval: apply decay by a factor of opt_decaybase every opt_ndecayepochs epochs

# Plotting and logging parameters
log_to_file = True         # set to True to output log information to a file rather than the console
logging_lvl = logging.INFO  # logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
plt_show = False       # show plots on-screen for dnnplot
plt_imgtype = "png"    # image type to which to save plots

# END USER INPUTS
# ------------------------------------------------------------------------------------------

# Calculated parameters (based on user inputs)
batches_per_epoch = int(2*dtblk_size/batch_size)
num_epoch_blks = num_epochs / epoch_blk_size
num_dt_blks = ntrain_evts / dtblk_size
pdim = int(vox_ext / vox_size)
npix = pdim * pdim
if(use_proj): nchannels = 3
if(use_3d):
    nchannels = 1
    npix = pdim * pdim * pdim
ch_blk = ((1.0*vox_ext)/vox_size) / nchannels
