from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np
import collections as cl
import tensorflow.keras.backend as K


class ScaledDotProductAttention(tf.keras.layersLayer):
  def _init_(self, masking=True, future=False, dropout_ratte=0., **kwargs):
    self._masking = masking
    self. future = future
    self._dropout_rate = dropout_rate
    self. masking_num = -2**32+1
    super(ScaledDotProductAttention, self).__init_(**kwargs)

  def mask(self, inputs, masks):
    masks = tf.keras.backend.cast(masks, 'float32')
    masks = tf.keras.backend.tile(masks, [tf.keras.shape(inputs)[0] // tf.keras.shape(masks)[0], 1])
    masks = tf.keras.backend.expand_dims(masks, 1)
    outputs = inputs + masks * self._masking_num
    return outputs

  def future_mask(self, inputs):
    diag_vals = tf.ones_like(inputs[0, :, :])
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_deense
    future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputss)[0], 1,1]
    paddings = tf.ones_like(future_masks) * self._masking_num
    outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs
    return outputs

  def call(self, inputs):
    if self. masking:
      assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
      queries, keys, values, masks = inputs
    else:
      assert len(inputs) = 3, "inputs should be set [queries, keyss,values].
      queries, keys, values = inputs
      
    if tf.keras.backend.dtype(queries) != 'float32': queries = tf.keras.backend.cast(queries, 'float32')
    if tf.keras.backend.dtype(keys) != 'float32': keys = tf.keras.backend.cast(keys, 'float32')    
    if tf.keras.backend.dtype(values) != 'float32': values = tf.keras.backend.cast(values, 'float32')

    matmul = tf.keras.backend.batch_dot(queries, tf.trannspose(keys,[0,2,1])
    scaled_matmul = matmul / int(keys.shape[-1]) ** 0.5
    
    if self.masking:
      scaled_matmul = self.mask(scaled_matmul, masks
    if self.future:
      scaled_matmul = self.future_mask(scaled_matmul)
      
    softmax_out = tf.nn.softmax(scaled_matmul,axis=-1)  #(1024*head)*t1*t2
    
    # Dropout
    out = tf.keras.backend.dropout(softmax_out, self._dropout_rate)
    outputs = tf.keras.backend.batch_dot(out,values) # (1024*head)*t1*t2
    
    return outputs,softmax_out  #(1024*head)*t1*h 1024*t1*tf2

    def compute_output_shape(self, input_shape):
      return input_shape

class MultiHeadAttention(tf.keras.layers.Layer):
  def _init_(self, n_heads, head_dim, dropout_rate=.1, masking=Truefuture=False, trainable=True, **kwargs):
    super(MultiHeadAttention, self)._init_(**kwargs)
    self._n_heads = n_heads
    self. head dim = head dim
    self._dropout_rate = dropout_rate
    self. masking = masking
    self. future = future
    self. trainable = trainable
    embed dim = n_heads * head dim
    self.wq = tf.keras.layers.Dense(embed_dim)
    self.wk = tf.keras.layers.Dense(embed_dim)    
    self.wv = tf.keras.layers.Dense(embed_dim)    
    self.dense =tf.keras.layers.Dense(embed_dim)
    
  def build(self, input_shape):
    super(MultiHeadAttention, self).build(input_shape）
                                          
  def split_heads(self, x):
    x = tf.concat(tf.split(x, self._n_heads, axis=-1)axis=0)
    return x

  def call(self, inputs):
    if self. masking:
      assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]
      queries, keys, values, masks = inputs
    else:
      assert len(inputs) == 3, "inputs should be set [queries, keys, vaalues]
      queries_linear, keys_linear, values_linear =  inputs

    q = self.wq(queries_linear)
    k = self.wk(keys_linear)
    v = self.wv(values_linear)

    queries_multi_heads= self.split_heads(q)
    keys_multi_heads = self.split_heads(k)
    values_multi_heads = self.split_heads(v)

    if self._masking:
      att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heaads, masks]
    else:
      att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

    attention = ScaledDotProductAttention(masking=self._masking, future=self._future, dropout_rate=self_dropout_rate)
    att_out,softmax_out = attention(att_inputs)   #(8*h)*T1*e(B*h)*T1*T2
    att_out = tf.concat(tf.unstack( tf.concat(tf.split(att_out, self._n_heads, axis=0),axis=-1),axis=1),axis=-1)
    softmax_out = tf.concat(tf.unstack( tf.concat(tf.split(sooftmax_out, self._n_heads, axis=0),axis=-1),axis=1),axis=-1)

    outputs = att_out  #B*(T1*H)
    softmax_out = softmax_out #B*(T1*T2*h)
    return outputs, softmax_out

  def compute_output_shape(self, input_shape):
    return input_shape


class TDHAM_Net(tf.keras.Model):
  def _init_(self, name='TDHAM_Net"," kwargs):
    super(TDHAM_Net, self).__init_(name=name, **kwargs)
             
    self.cur hh_embed_layer = tf.keras.layers.DensseFeatures(cur_hh_embed_columns, name="cur_hh_embed_layer")
    self.fin_layer = tf.keras.layers.Densefeatures(fin_coluumns, name = "fin_layer")
    self.user_layer = tf.keras.layers.DenseFeatures(user_columns, name="user_layer")
    self.dense_layer_usr_fin = tf.keras.layers.Dense(16activation='relu')

    self.deep_layer = tf.keras.layers.Dense(16,activation='relu')
    self.deep_layer_up = tf.keras.layers.Dense(16,activation='relu')
    self.out_layer = tf.keras.layers.Dense(1,activation='sigmoi')
    self.out_layer_up = tf.keras.layers.Dense(1,activation='sigmpid')
    self.out_layer_hh_up = tf.keras.layers.Dense(1,activattion='sigmoid',name="hh up")

    self.att_layer = ScaledDotProductAttention(masking=False,dropout_rate=0.0)


  def call(self, inputs):
    cur_hh_embed_layer = self.cur_hh_embed_layer(inputs)

    




  

    
