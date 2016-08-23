"""
dnntrain.py

Trains the DNN analysis for the specified configuration

"""

# Temporary fix to force use of user-installed modules
#  to correct error using out-of-date "six" module which
#  was not allowing for import of tensorflow
import sys
sys.path.insert(0,sys.path[4])

import h5py
import numpy as np
import tensorflow as tf
import os
import logging
import gc

import nets.neuralnets
from nets.gnet import GoogleNet
from dnninputs import *

# Ensure the appropriate directory structure exists.
if(not os.path.isdir(rdir)): os.mkdir(rdir)
if(not os.path.isdir("{0}/{1}".format(rdir,rname))): os.mkdir("{0}/{1}".format(rdir,rname))
if(not os.path.isdir("{0}/{1}/acc".format(rdir,rname))): os.mkdir("{0}/{1}/acc".format(rdir,rname))

# Create the logger object.
if(log_to_file):
    logging.basicConfig(filename="{0}/{1}/{2}.log".format(rdir,rname,rname),format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)
logging.info("Params:\n ntrain_evts = {0}\n num_epochs = {1}\n epoch_blk_size = {2}\n dtblk_size = {3}\n batch_size = {4}\n nval_evts = {5}\n opt_lr = {6}\n opt_eps = {7}\n opt_decaybase = {8}\n opt_ndecayepochs = {9}\n opt_mom = {10}".format(ntrain_evts,num_epochs,epoch_blk_size,dtblk_size,batch_size,nval_evts,opt_lr,opt_eps,opt_decaybase,opt_ndecayepochs,opt_mom))

# Checks on parameters.
if(ntrain_evts % dtblk_size != 0):
    logging.error("ERROR: ntrain_evts must be evenly divisible by dtblk_size..."); exit()
if(num_epochs % epoch_blk_size != 0):
    logging.error("ERROR: num_epochs must be evenly divisible by epoch_blk_size..."); exit()
if(ntrain_evts % batch_size != 0):
    logging.error("ERROR: ntrain_evts must be evenly divisible by batch_size..."); exit()
if(nval_evts % batch_size != 0):
    logging.error("ERROR: nval_evts must be evenly divisible by batch_size..."); exit()

# Constructed file names.
fname_si = "{0}/{1}_si.h5".format(datdir,dname)
fname_bg = "{0}/{1}_bg.h5".format(datdir,dname)
fn_saver = "{0}/{1}/tfmdl_{2}.ckpt".format(rdir,rname,rname)   # for saving trained network
fn_acc = "{0}/{1}/acc/accuracy_{2}.dat".format(rdir,rname,rname)
fn_prob = "{0}/{1}/acc/prob_{2}".format(rdir,rname,rname)

# ---------------------------------------------------------------------------------------
# Function definitions
# ---------------------------------------------------------------------------------------

# Evaluate the performance.
def eval_performance(fsummary,epoch,sess,loss,y_out,dat_train_si,dat_train_bg,dat_test_si,dat_test_bg):

    logging.info(" \n --- Calling eval_performance\n")

    # ----------------------------
    # Evaluate the training data.
    # ----------------------------
    f_prob_train = open("{0}_train_ep{1}.dat".format(fn_prob,epoch),"w")
    acc_tr_si = 0.; acc_tr_bg = 0.; lval_tr_si = 0.; lval_tr_bg = 0.
    # Signal
    #print "-- TRAINING data: signal"
    nevt = 0; nbatches = 0
    while(nevt < nval_evts):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train_si[nevt:nevt+batch_size], y_: lbl_train_si[nevt:nevt+batch_size]})
        for y0 in ytemp[:,0]:
            if(y0 > 0.5): acc_tr_si += 1
        lval_tr_si += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_train.write("{0} {1} {2}\n".format(1,y0,y1))
        nevt += batch_size; nbatches += 1

    # Background
    #print "-- TRAINING data: background"
    nevt = 0; nbatches = 0
    while(nevt < nval_evts):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train_bg[nevt:nevt+batch_size], y_: lbl_train_bg[nevt:nevt+batch_size]})
        for y1 in ytemp[:,1]:
            if(y1 > 0.5): acc_tr_bg += 1
        lval_tr_bg += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_train.write("{0} {1} {2}\n".format(0,y0,y1))
        nevt += batch_size; nbatches += 1
    f_prob_train.close()

    acc_tr_si /= nval_evts; acc_tr_bg /= nval_evts
    lval_tr_si /= nbatches; lval_tr_bg /= nbatches

    # ---------------------------
    # Evaluate the test data.
    # ---------------------------
    f_prob_test = open("{0}_test_ep{1}.dat".format(fn_prob,epoch),"w")
    acc_te_si = 0.; acc_te_bg = 0.; lval_te_si = 0.; lval_te_bg = 0.
    # Signal
    #print "-- TEST data: signal"
    nevt = 0; nbatches = 0
    while(nevt < nval_evts):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_test_si[nevt:nevt+batch_size], y_: lbl_test_si[nevt:nevt+batch_size]})
        for y0 in ytemp[:,0]:
            if(y0 > 0.5): acc_te_si += 1
        lval_te_si += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_test.write("{0} {1} {2}\n".format(1,y0,y1))
        nevt += batch_size; nbatches += 1

    # Background
    #print "-- TEST data: background"
    nevt = 0; nbatches = 0
    while(nevt < nval_evts):
        ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_test_bg[nevt:nevt+batch_size], y_: lbl_test_bg[nevt:nevt+batch_size]})
        for y1 in ytemp[:,1]:
            if(y1 > 0.5): acc_te_bg += 1
        lval_te_bg += ltemp
        for y0,y1 in zip(ytemp[:,0],ytemp[:,1]): f_prob_test.write("{0} {1} {2}\n".format(0,y0,y1))
        nevt += batch_size; nbatches += 1
    f_prob_test.close()

    acc_te_si /= nval_evts; acc_te_bg /= nval_evts
    lval_te_si /= nbatches; lval_te_bg /= nbatches

    # Write to the final summary file.
    fsummary.write("{0} {1} {2} {3} {4} {5} {6} {7} {8}\n".format(epoch,acc_tr_si,acc_tr_bg,lval_tr_si,lval_tr_bg,acc_te_si,acc_te_bg,lval_te_si,lval_te_bg))

# Read in all the data from evt_start to evt_end-1.
# - assumes that events are labeled as trkX in the hdf5 file, where X is the event number
def read_data(h5f_si,h5f_bg,evt_start,evt_end):

    nevts = evt_end - evt_start

    logging.info("-- read_data: Reading events from {0} to {1} for signal and background with {2} channels".format(evt_start,evt_end,nchannels))

    # Set up the data arrays.
    dat_si = np.zeros([nevts,nchannels*npix]); lbl_si = np.zeros([nevts,lbl_size])
    dat_bg = np.zeros([nevts,nchannels*npix]); lbl_bg = np.zeros([nevts,lbl_size])

    # Read in all events from the hdf5 files.
    ntrk = 0
    while(ntrk < nevts):

        trkn_si = h5f_si['trk{0}'.format(evt_start+ntrk)]
        trkn_bg = h5f_bg['trk{0}'.format(evt_start+ntrk)]

        if(use_3d):

            if(ntrk == 0): print "Using 3D data..."

            # Read the signal event.
            xarr = trkn_si[0]; yarr = trkn_si[1]; zarr = trkn_si[2]; earr = trkn_si[3]
            for xx,yy,zz,ee in zip(xarr,yarr,zarr,earr): 
                dat_si[ntrk][int(zz*pdim*pdim + yy*pdim + xx)] += ee
            dat_si[ntrk] *= vox_norm/max(dat_si[ntrk])
            lbl_si[ntrk][0] = 1    # set the label to signal

            # Read the background event.
            xarr = trkn_bg[0]; yarr = trkn_bg[1]; zarr = trkn_bg[2]; earr = trkn_bg[3]
            for xx,yy,zz,ee in zip(xarr,yarr,zarr,earr): 
                dat_bg[ntrk][int(zz*pdim*pdim + yy*pdim + xx)] += ee
            dat_bg[ntrk] *= vox_norm/max(dat_bg[ntrk])
            lbl_bg[ntrk][1] = 1    # set the label to signal

        else:

            if(use_proj):

                if(ntrk == 0): print "Creating 3 projections from data..."
    
                # Read the signal event.
                xarr = trkn_si[0]; yarr = trkn_si[1]; zarr = trkn_si[2]; earr = trkn_si[3]
                for xx,yy,ee in zip(xarr,yarr,earr): dat_si[ntrk][3*int(yy*pdim + xx)] += ee         # x-y projection
                for yy,zz,ee in zip(yarr,zarr,earr): dat_si[ntrk][3*int(zz*pdim + yy) + 1] += ee     # y-z projection
                for xx,zz,ee in zip(xarr,zarr,earr): dat_si[ntrk][3*int(zz*pdim + xx) + 2] += ee     # x-z projection
                dat_si[ntrk] *= vox_norm/max(dat_si[ntrk])
                lbl_si[ntrk][0] = 1    # set the label to signal
                if(lbl_size == 6):
                    lbl_si[ntrk][2] = 1
                    lbl_si[ntrk][4] = 1
        
                # Read the background event.
                xarr = trkn_bg[0]; yarr = trkn_bg[1]; zarr = trkn_bg[2]; earr = trkn_bg[3]
                for xx,yy,ee in zip(xarr,yarr,earr): dat_bg[ntrk][3*int(yy*pdim + xx)] += ee         # x-y projection
                for yy,zz,ee in zip(yarr,zarr,earr): dat_bg[ntrk][3*int(zz*pdim + yy) + 1] += ee     # y-z projection
                for xx,zz,ee in zip(xarr,zarr,earr): dat_bg[ntrk][3*int(zz*pdim + xx) + 2] += ee     # x-z projection
                dat_bg[ntrk] *= vox_norm/max(dat_bg[ntrk])
                lbl_bg[ntrk][1] = 1    # set the label to background
                if(lbl_size == 6):
                    lbl_bg[ntrk][3] = 1
                    lbl_bg[ntrk][5] = 1
    
            else:

                if(ntrk == 0): print "Creating {0} z-slices from data...".format(nchannels)
    
                # Read the signal event.
                xarr = trkn_si[0]; yarr = trkn_si[1]; zarr = trkn_si[2]; earr = trkn_si[3]
                for xx,yy,zz,ee in zip(xarr,yarr,zarr,earr):
                   
                    # Extract each channel
                    ch = int(zz/ch_blk)
                    dat_si[ntrk][nchannels*int(yy*pdim + xx) + ch] += ee
           lbl_size        
                dat_si[ntrk] *= vox_norm/max(dat_si[ntrk])
                lbl_si[ntrk][0] = 1    # set the label to signal
    
                # Read the background event.
                xarr = trkn_bg[0]; yarr = trkn_bg[1]; zarr = trkn_bg[2]; earr = trkn_bg[3]
                for xx,yy,zz,ee in zip(xarr,yarr,zarr,earr):
    
                    # Extract each channel
                    ch = int(zz/ch_blk)
                    dat_bg[ntrk][nchannels*int(yy*pdim + xx) + ch] += ee
    
                dat_bg[ntrk] *= vox_norm/max(dat_bg[ntrk])
                lbl_bg[ntrk][1] = 1    # set the label to signal

        ntrk += 1
        if(ntrk % 1000 == 0):
            logging.info("-- Read {0} events...".format(ntrk))

    # Normalize each event.
    for sievt in dat_si:
        sievt[:] = sievt[:] - np.mean(sievt)
        sievt[:] = sievt[:]/np.std(sievt)

    for bgevt in dat_bg:
        bgevt[:] = bgevt[:] - np.mean(bgevt)
        bgevt[:] = bgevt[:]/np.std(bgevt)

    # Return the data and labels.
    return (dat_si,lbl_si,dat_bg,lbl_bg)

# Set up the neural network.
def net_setup():

    logging.info("\n\n-- net_setup():  SETTING UP NETWORK --")

    logging.info("Creating placeholders for input and output variables...")
    x_input = tf.placeholder(tf.float32, [batch_size, nchannels*npix]) # npix])
    y_ = tf.placeholder(tf.float32, [batch_size, 2])

    # Set up the GoogleNet
    if(read_googlenet):
        x_image = tf.reshape(x_input, [-1,pdim,pdim,3])
        logging.info("Reading in GoogLeNet model...")
        net = GoogleNet({'data':x_image})
        y_out = net.get_output()
    else:
       y_out = nets.neuralnets.getNet(net_name,x_input) 

    # Set up for training
    logging.info("Setting up tf training variables...")
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_out + 1.0e-9))
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    gstep = tf.Variable(0, trainable=False)
    lrate = tf.train.exponential_decay(opt_lr, gstep,
                                           opt_ndecayepochs*batches_per_epoch, opt_decaybase, staircase=True)
    #train_step = tf.train.MomentumOptimizer(learning_rate=opt_lr,momentum=opt_mom).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=lrate,epsilon=opt_eps).minimize(cross_entropy,global_step=gstep)
    #train_step = tf.train.GradientDescentOptimizer(lrate).minimize(cross_entropy,global_step=gstep)

    logging.info("Setting up session...")
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    # Create a saver to save the DNN.
    saver = tf.train.Saver()

    # Load in the previously trained data.
    if(read_googlenet and train_init):
        logging.info("NOT Loading in previously trained GoogLeNet parameters...")
        net.load('nets/params/gnet.npy', sess)
    elif(not train_init):
        logging.info("Restoring previously trained net from file {0}".format(fn_saver))
        saver.restore(sess,fn_saver) 

    return (sess,train_step,loss,x_input,y_,y_out,saver)

# -----------------------------------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------------------------------

# Set up the DNN.
(sess,train_step,loss,x,y_,y_out,saver) = net_setup()

# Open the relevant files.
f_acc = open(fn_acc,'w')
h5f_si = h5py.File(fname_si,'r')
h5f_bg = h5py.File(fname_bg,'r')

# Read in a validation set for short checks on accuracy.
dat_val_si = np.zeros([batch_size,nchannels*npix]); lbl_val_si = np.zeros([batch_size,lbl_size])
dat_val_bg = np.zeros([batch_size,nchannels*npix]); lbl_val_bg = np.zeros([batch_size,lbl_size])
(dat_val_si[:],lbl_val_si[:],dat_val_bg[:],lbl_val_bg[:]) = read_data(h5f_si,h5f_bg,ntrain_evts,ntrain_evts+batch_size)

# Set up the arrays for the training and validation evaluation datasets.
dat_train_si = []; lbl_train_si = []; dat_train_bg = []; lbl_train_bg = []
dat_test_si = []; lbl_test_si = []; dat_test_bg = []; lbl_test_bg = []

# Iterate over all epoch blocks.
for eblk in range(num_epoch_blks):

    logging.info("\n\n**EPOCH BLOCK {0}".format(eblk))

    # Iterate over data blocks.
    for dtblk in range(num_dt_blks):

        logging.info("- DATA BLOCK {0}".format(dtblk))

        if(num_dt_blks > 1 or eblk == 0):

            # Read in the data.
            evt_start = dtblk*dtblk_size
            evt_end = (dtblk+1)*dtblk_size
            dat_train = np.zeros([2*dtblk_size,nchannels*npix])
            lbl_train = np.zeros([2*dtblk_size,lbl_size])
            gc.collect()  # force garbage collection to free memory
            (dat_train[0:dtblk_size],lbl_train[0:dtblk_size],dat_train[dtblk_size:],lbl_train[dtblk_size:]) = read_data(h5f_si,h5f_bg,evt_start,evt_end)

        # Iterate over epochs within the block.
        for ep in range(epoch_blk_size):

            logging.info("-- EPOCH {0} of block size {1}".format(ep,epoch_blk_size))

            # Shuffle the data.
            logging.info("--- Shuffling data...")
            perm = np.arange(len(dat_train))
            np.random.shuffle(perm)
            #dat_train = dat_train[perm]
            #lbl_train = lbl_train[perm]

            # Train the NN in batches.
            for bnum in range(batches_per_epoch):

                logging.info("--- Training batch {0} of {1}".format(bnum,batches_per_epoch))

                batch_xs = dat_train[perm[bnum*batch_size:(bnum + 1)*batch_size],:]
                batch_ys = lbl_train[perm[bnum*batch_size:(bnum + 1)*batch_size],:]
                _, loss_val = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
                logging.info("--- Got loss value of {0}".format(loss_val))

            # Run a short accuracy check.
            acc_train = 0.; acc_test_si = 0.; acc_test_bg = 0.
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_train[perm[0:batch_size]], y_: lbl_train[perm[0:batch_size]]})
            for yin,yout in zip(lbl_train[perm[0:batch_size]],ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_train += 1
            acc_train /= batch_size
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_val_si, y_: lbl_val_si})
            for yin,yout in zip(lbl_val_si,ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_test_si += 1
            acc_test_si /= batch_size
            ltemp,ytemp = sess.run([loss,y_out],feed_dict={x: dat_val_bg, y_: lbl_val_bg})
            for yin,yout in zip(lbl_val_bg,ytemp):
                if(np.argmax(yin) == np.argmax(yout)): acc_test_bg += 1
            acc_test_bg /= batch_size
            logging.info("--- Training accuracy = {0}; Test signal accuracy = {1}, Test background accuracy = {2}".format(acc_train,acc_test_si,acc_test_bg))

    # Calculate the number of epochs run.
    epoch = eblk*epoch_blk_size
    logging.info("Checking accuracy after {0} epochs".format(epoch+1))

    # Read in the data to be used in the accuracy check.
    if(len(dat_train_si) == 0):
        dat_train_si = np.zeros([nval_evts,nchannels*npix]); lbl_train_si = np.zeros([nval_evts,lbl_size])
        dat_train_bg = np.zeros([nval_evts,nchannels*npix]); lbl_train_bg = np.zeros([nval_evts,lbl_size])
        (dat_train_si[:],lbl_train_si[:],dat_train_bg[:],lbl_train_bg[:]) = read_data(h5f_si,h5f_bg,0,nval_evts)

    if(len(dat_test_si) == 0):
        dat_test_si = np.zeros([nval_evts,nchannels*npix]); lbl_test_si = np.zeros([nval_evts,lbl_size])
        dat_test_bg = np.zeros([nval_evts,nchannels*npix]); lbl_test_bg = np.zeros([nval_evts,lbl_size])
        (dat_test_si[:],lbl_test_si[:],dat_test_bg[:],lbl_test_bg[:]) = read_data(h5f_si,h5f_bg,ntrain_evts,ntrain_evts+nval_evts)

    # Run the accuracy check.
    eval_performance(f_acc,epoch,sess,loss,y_out,dat_train_si,dat_train_bg,dat_test_si,dat_test_bg)

    # Save the trained model every nepoch_save epochs.
    if(eblk % nepoch_save == 0): 
        logging.info("Saving trained model to: {0}".format(fn_saver))
        save_path = saver.save(sess, fn_saver)

# Close the relevant files.
f_acc.close()
h5f_si.close()
h5f_bg.close()
