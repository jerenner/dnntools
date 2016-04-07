import logging

# ---------------------------------------------------------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------------------------------------------------------

# Directory and run names
datdir = "/home/lsantilli/data"                          # the base data directory
dname = "vox_dnn3d_NEXT100_Paolina222_v10x10x10_r200x200x200" # the data name
rdir = "/home/lsantilli/dnn/run3"                             # the run directory
rname = "dnn_10x10x10_new_gnl_2"     # the run name

# Net configuration parameters
net_name = "MNISTnew"                    # name of the neural net described in nets/neuralnets.py
read_googlenet = False                        # set to true only if using the GoogLeNet
train_init = True                             # if true, train from net with standard pre-training; if false, read in a previously trained net

# Voxel parameters
vox_ext = 200          # extent of voxelized region in 1 dimension (in mm)
vox_size = 10           # voxel size (in mm)
vox_norm = 1.0         # voxel normalization

# Parameters describing training intervals and number of events for training and validation
ntrain_evts = 40000    # number of training evts per dataset
nval_evts = 9280       # number of validation events
num_epochs = 70        # total number of epochs to train
epoch_blk_size = 1     # number of epochs to run per block (before reading new dataset); set equal to num_epochs unless data to be read in multiple blocks
dtblk_size = 40000      # number of signal and background events per training block
batch_size = 320       # training batch size

# Training optimizer parameters
opt_lr = 1.0e-4        # optimizer learning rate
opt_eps = 1.0e-6       # optimizer epsilon (for AdamOptimizer)
opt_mom = 0.9          # optimizer momentum
opt_decaybase = 0.1    # multiplicative factor for learning rate decay
opt_ndecayepochs = 10  # decay interval: apply decay by a factor of opt_decaybase every opt_ndecayepochs epochs

# Plotting and logging parameters
log_to_file = True         # set to True to output log information to a file rather than the console
logging_lvl = logging.INFO  # logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
plt_show = False       # show plots on-screen for dnnplot

# END USER INPUTS
# ------------------------------------------------------------------------------------------

# Calculated parameters (based on user inputs)
batches_per_epoch = int(2*dtblk_size/batch_size)
num_epoch_blks = num_epochs / epoch_blk_size
num_dt_blks = ntrain_evts / dtblk_size
pdim = int(vox_ext / vox_size)
npix = pdim * pdim
