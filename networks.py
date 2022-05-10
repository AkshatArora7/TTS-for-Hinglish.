from __future__ import print_function

from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse): 
        
        prenet_out = prenet(inputs, is_training=is_training) 
        
        
        enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training)
        
        
        enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")  
          
        ## Conv1D projections
        enc = conv1d(enc, filters=hp.embed_size//2, size=3, scope="conv1d_1") 
        enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        enc = conv1d(enc, filters=hp.embed_size // 2, size=3, scope="conv1d_2") 
        enc = bn(enc, is_training=is_training, scope="conv1d_2")

        enc += prenet_out # residual connections
          
        ## Highway Nets
        for i in range(hp.num_highwaynet_blocks):
            enc = highwaynet(enc, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) # (N, T_x, E/2)

        ## Bidirectional GRU
        memory = gru(enc, num_units=hp.embed_size//2, bidirection=True) # (N, T_x, E)
    
    return memory
        
def decoder1(inputs, memory, is_training=True, scope="decoder1", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = prenet(inputs, is_training=is_training) 

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=hp.embed_size) 

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru1") 
        dec += gru(dec, hp.embed_size, bidirection=False, scope="decoder_gru2") 
          
        
        mel_hats = tf.layers.dense(dec, hp.n_mels*hp.r)
    
    return mel_hats, alignments

def decoder2(inputs, is_training=True, scope="decoder2", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, hp.n_mels])

        dec = conv1d_banks(inputs, K=hp.decoder_num_banks, is_training=is_training)
         
        dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding="same")
        dec = conv1d(dec, filters=hp.embed_size // 2, size=3, scope="conv1d_1") 
        dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

        dec = conv1d(dec, filters=hp.n_mels, size=3, scope="conv1d_2")  
        dec = bn(dec, is_training=is_training, scope="conv1d_2")

        # Extra affine transformation for dimensionality sync
        dec = tf.layers.dense(dec, hp.embed_size//2) 
         
        # Highway Nets
        for i in range(4):
            dec = highwaynet(dec, num_units=hp.embed_size//2, 
                                 scope='highwaynet_{}'.format(i)) 
         
        # Bidirectional GRU    
        dec = gru(dec, hp.embed_size//2, bidirection=True)
        outputs = tf.layers.dense(dec, 1+hp.n_fft//2)

    return outputs
