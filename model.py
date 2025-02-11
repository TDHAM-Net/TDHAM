from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np
import collections as cl
import tensorflow.keras.backend as K

import json,sys,os,time

hour_num = 8
hhlist = [9,10,11,12,14,15,16,17]
conf_file = sys.argv[1]
with open(conf_file, 'r') as f:
    data=json.load(f)

column_names =data['column_names']
feature_name = data['feature_name']
label_name = data['label_name']
time_feature_name = data['time_feature_name']
no_time_feature_name = data['no_time_feature_name']
user_profile_name = data['user_profile_name']
CATEGORIES = data['CATEGORIES']
CATEGORIES_SIZE = data['CATEGORIES_SIZE']

batch_size=data['batch_size']
train_file = data['train_file']

in_var_columns = []
for i in range(hour_num):
    in_var_columns.append([])
time_columns=[]
no_time_columns = []
user_columns = []
hour_column = []
for feature in feature_name:
    if feature == "call_hour":
        ft = tf.feature_column.numeric_column(feature)
        hour_column.append(ft)
    if(feature in CATEGORIES.keys()):
        cal_col = tf.feature_column.categorical_column_with_identity(keyfeature, num_buckets-CATEGORIES[feature])
        ft = tf.feature_column.embedding_column(cal_col, dimension-CATEGORIES_SIZE[feature])
    else:
        ft = tf.feature_column.numeric_column(feature)
    if feature in  user_profile_name:
        user_columns.append(ft)
    else:
        if feature in time_feature_name:
            time_columns.append(ft)
        if feature in no_time_feature_name:
            h = (feature.split("_")[2]).split("h")[0]
            idx =  hhlist.index(int(h))
            in_var_columns[idx].append(ft)
            no_time_columns.append(ft)


def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size = batch_size,
    column_names=column_names,
    label_name=label_name,
    prefetch_buffer_size=batch_size,
    num_parallel_reads = 4,
    num_epochs=1
    )
return dataset


def pack_features_vector(features, labels):
    feature["call_hour"] = tf.cast(feature["call_hour"], "int32")
    feature["call_hour"] = tf.where(feature["call_hour"]>13, feature["call_hour"]-10,feature["call_hour"]-9)

    output = cl.OrderedDict()
    output[label_name] = labels
    output["pre_y1"] = labels
    output["tss"] = labels * 0
return features,output

train_dataset = get_dataset(train_file)
train_dataset = train_dataset.map(pack_features_vector)

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
    future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputss)[0], 1,1])
    paddings = tf.ones_like(future_masks) * self._masking_num
    outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
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

    matmul = tf.keras.backend.batch_dot(queries, tf.trannspose(keys,[0,2,1]))
    scaled_matmul = matmul / int(keys.shape[-1]) ** 0.5
    
    if self.masking:
      scaled_matmul = self.mask(scaled_matmul, masks)
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
    self.head_dim = head_dim
    self.dropout_rate = dropout_rate
    self.masking = masking
    self.future = future
    self.trainable = trainable
    embed_dim = n_heads * head_dim
    self.wq = tf.keras.layers.Dense(embed_dim)
    self.wk = tf.keras.layers.Dense(embed_dim)    
    self.wv = tf.keras.layers.Dense(embed_dim)    
    self.dense =tf.keras.layers.Dense(embed_dim)
    
  def build(self, input_shape):
    super(MultiHeadAttention, self).build( input_shape ）
                                          
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
  def _init_(self, name="TDHAM_Net", **kwargs):
    super(TDHAM_Net, self).__init_(name=name, **kwargs)
             
    self.cur_hh_embed_layer = tf.keras.layers.DensseFeatures(time_columns, name="cur_hh_embed_layer")
    self.non_cur_hh_embed_layer = tf.keras.layers.Densefeatures(no_time_columns, name = "fin_layer")
    self.user_layer = tf.keras.layers.DenseFeatures(user_columns, name="user_layer")
    self.hh_value_layer = tf.keras.layers.DenseFeatures(hour_column, name="user_layer")

    self.in_var_embed_layer=[]
    for v in in_var_columns:
        self.in_var_embed_layer.append(tf.keras.layers.DenseFeatures(v))

    self.dense_layer_usr_fin = tf.keras.layers.Dense(16, activation='relu')

    self.deep_layer_tpe1 = tf.keras.layers.Dense(16,activation='relu')
    self.deep_layer_tpe2 = tf.keras.layers.Dense(16,activation='relu')
    self.deep_layer_tss = tf.keras.layers.Dense(16,activation='relu')
    self.deep_layer_idd1 = tf.keras.layers.Dense(16,activation='relu')
    self.deep_layer_idd2 = tf.keras.layers.Dense(16,activation='relu')

    self.out_layer_tpe = tf.keras.layers.Dense(1,activation='sigmoid')
    self.out_layer_tss = tf.keras.layers.Dense(1,activattion='sigmoid')
    self.out_layer_idd = tf.keras.layers.Dense(1,activattion='sigmoid')

    self.self_att_layer = MultiHeadAttention(8, 16, masking=False,dropout_rate=0.0)
    self.cross_att_layer = MultiHeadAttention(8, 16, masking=False,dropout_rate=0.0)

    self.linear_layer = tf.keras.layers.Dense(1,activation='softmax') 


  def call(self, inputs):
    cur_hh_embed_layer = self.cur_hh_embed_layer(inputs)
    non_cur_hh_embed_layer = self.non_cur_hh_embed_layer(inputs)
    user_layer = self.user_layer(inputs)

    in_var_embed_layer=[]
    for v in self.in_var_embed_layer:
        in_var_embed_layer.append(tf.expand_dims(v(inputs), axis=1))
    in_var_embed_layer = tf.concat(in_var_embed_layer, axis=1)
    in_var_att, _ = self.self_att_layer([non_cur_hh_embed_layer, in_var_embed_layer, in_var_embed_layer])
    tp_a = self.deep_layer_tpe1(tf.concat(tf.unstack(in_var_att, axis=1), axis=1))
    tp_a = self.out_layer_tpe(tp_a)
    
    tp_p = tf.softmax(self.deep_layer_tpe2(in_var_att))
    pre_y1 = tp_a - 1/hour_num + tp_p

    usr_fin_layer = self.dense_layer_usr_fin(tf.concat([user_layer, in_var_att,cur_hh_embed_layer], axis=1))

    t_q=tf.expand_dims(usr_fin_layer, axis=1)
    t_k_v=tf.concat(
            [tf.expand_dims(usr_fin_layer, axis=1)
            ,tf.expand_dims(cur_hh_embed_layer, axis=1)
            ,tf.expand_dims(in_var_att, axis=1)]
        ,axis=1)

    att_v, att_w = self.cross_att_layer(t_q, t_k_v, t_k_v)
    att_v = self.deep_layer_idd1(tf.concat(tf.unstack(att_v, axis=1), axis=1))
    att_v = self.out_layer_tpe(att_v)
    
    idd_t = tf.softmax(self.deep_layer_idd2(tf.concat(tf.unstack(att_w, axis=1), axis=1)))

    pre_y2 = att_v*idd_t

    cur_hh_idx = self.hh_value_layer(inputs)
    pre_y1 = tf.gather(pre_y1, cur_hh_idx)
    pre_y2 = tf.gather(pre_y2, cur_hh_idx)

    tss_y = self.deep_layer_tss(tf.concat(tf.unstack(att_w, axis=1), axis=1))
    sigma_1 = self.out_layer_tss(tss_y)
    sigma_2 = self.linear_layer(pre_y1 - tp_a)
    tss_y = sigma_1 - sigma_2

    outputs = cl.OrderedDict()
    outputs["pre_y1"] = pre_y1
    outputs[label_name] = pre_y2
    outputs["tss"] = tss_y

    return outputs

model = TDHAM_Net()

loss_dict = {}
loss_weights_dict={}
metrics_dict={}
if(True):
    loss_dict[label_name]= 'binary_crossentropy'
    loss_weights_dict[label_name] =1.0
    metrics_dict[label_name] = ['accuracy', 'AUC']

    loss_dict["pre_y1"]= 'binary_crossentropy'
    loss_weights_dict["pre_y1"] =1.0
    metrics_dict["pre_y1"] = ['AUC']

    loss_dict["tss"]= 'mse'
    loss_weights_dict["tss"] =1.0
    metrics_dict["tss"] = ['mae']

model.compile(
    loss=loss_dict,
    loss_weights=loss_weights_dict,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=metrics_dict
    )

history1 = model.fit(train_dataset, batch_size=batch_size, epochs=1, verbose=1) # validation_data=valid_dataset,
model.save("./checkpoints/save_model", ssave_format='tf')


    
