
from __future__ import print_function

from hyperparams import Hyperparams as hp
import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)


def bn(inputs,
       is_training=True,
       activation_fn=None,
       scope="bn",
       reuse=None):
    
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               fused=True,
                                               reuse=reuse)
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:  # fallback to naive batch norm
        outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs

def conv1d(inputs, 
           filters=None, 
           size=1, 
           rate=1, 
           padding="SAME", 
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        
        if filters is None:
            filters = inputs.get_shape().as_list[-1]
        
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn, 
                "use_bias":use_bias, "reuse":reuse}
        
        outputs = tf.layers.conv1d(**params)
    return outputs

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, hp.embed_size//2, 1) # k=1
        for k in range(2, K+1): # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, hp.embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs # (N, T, Hp.embed_size//2*K)

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
            
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs

def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, 
                                                                   memory)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                                  attention_mechanism,
                                                                  num_units,
                                                                  alignment_history=True)
        outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs, dtype=tf.float32) 

    return outputs, state

def prenet(inputs, num_units=None, is_training=True, scope="prenet", reuse=None):
    
    if num_units is None:
        num_units = [hp.embed_size, hp.embed_size//2]
        
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate, training=is_training, name="dropout2") 
    return outputs # (N, ..., num_units[1])

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H*T + inputs*(1.-T)
    return outputs
