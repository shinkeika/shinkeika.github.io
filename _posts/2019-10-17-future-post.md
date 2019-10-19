---
title: 'Keras 极简教程'
date: 2019-10-17
permalink: /posts/2019/10/blog-post-1/
tags:
  - cool posts
  - category1
  - category2
---

Keras 极简教程 

#### 改backend

配置文件: ～/.keras/keras.json

类型 : theano, tensotflow 

1. 直接改文件，永久改动
2. os.environ['KERAS'_BACKEND] = 'theano' 临时改变



### 搭建回归神经网络

#### （Regressor）

```python
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(output_dim=(),input_dim=())) # 第一层需要定义
model.add(Dense(output_dim=())) # 默认第二层的输入就是第一层的输出
model.compile(loss='mse',optimizer='sgd')

# 训练
for step in range(1000):
	cost = model.train_on_batch(X_train,Y_train)
	if step % 100 == 0:
    print('train cost',cost)
# 评估   
cost = model.evaluate(X_test,Y_test,batch_size=40)
# 打印权重
W,b = model.layer[0].get_weights()
# 预测
Y_pred= model.predict(X_tests)
```

#### Classifier

```python
from keras.dataset import mnist
from keras.utils import np_utils  # 为了one-hot
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# method 1
model = Sequential([
  Dense(32, input_dim=784),  # 32 是 output_dim
  Activation('relu'),
  Dense(10), # 不用定义input ，input为上一层
  Activation('softmax')
])

rmsprop = RMSprop(lr=0.001, rho = 0.9, epsilon=1e-08, decay=0.0)

model.compile(
optimizer=rmsprop,  # 用默认的这样写optimizer='rmsprop'
loss = 'categorical_crossentropy',
metrics = ['accuracy'],
)
model.fit(X_train, y_train, nb_epochs=2, batch_size=32)
loss,accuracy = model.evaluate(X_test, y_test)
```

#### CNN

```python
from keras.dataset import mnist
from keras.utils import np_utils
from keras.model import Sequential
from keras.layers import Dense, Activation Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

model = Sequential()

model.add(Convolution2D(
	nb_filter=32, # 过滤器个数
  nb_row=5,  # 卷积核形状
  nb_col=5,  # 卷积核形状
  border_model = 'same',  # padding method
	input_shape(1,  # channels
              28,28)  # height $ width
))

model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape(32, 14, 14)
model.add(MaxPooling2D(
	pool_size(2,2),
  strides=(2,2),
  border_mode = 'same',
))
# Conv layer 2 output shape(64,14,14)
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape(64, 7, 7)
model.add(MaxPooling2D(
	pool_size(2,2),
  border_mode = 'same',
))

# Fully connected layer 1 input shape (64 * 7 *7 ) = (3136), output shape(1024)

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 shape(10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

model.compilt(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# train model
model.fit(X_train, y_train, nb_epoch=1,batch_size=32)

# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test,y_test)
```

#### RNN(classifier)

```python
from keras.utils import np_utils
from keras.model import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 28  # same as the height of the image
INPUT_SIZE = 28  # same as the width of the image
BATCH_SIZE = 50  
BATCH_INDEX = 0
OUTPUT_SIZE = 10  # one-hot分类
CELL_SIZE = 50  # RNN 的 hidden unit 个数
LR = 0.001

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
  batch_input_shape = (BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
	output_dim = CELL_SIZE
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(ACtivation('softmax'))  # 默认tanh

# optimizer
adam = Adam(LR)
model.compilt(optimizer=adam,
             loss = ' categorical_crossentropy',
             metrics = ['accuracy']
             )
# training
for step in range(40001):
  # data shape = (batch_num, steps, inputs/outputs)
	X_batch = X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:,:]
	Y_batch = y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX,:]

  cost = model.train_on_batch(X_batch, Y_batch)
  
  BATCH_INDEX += BATCH_SIZE
  BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
  if step % 500 == 0:
    cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0],verbose=False)
```

#### RNN(Regressor)

```python
from keras.utils import np_utils
from keras.model import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 20  # same as the height of the image
INPUT_SIZE = 1  # same as the width of the image
BATCH_SIZE = 50  
BATCH_INDEX = 0
OUTPUT_SIZE = 1  # one-hot分类
CELL_SIZE = 20  # RNN 的 hidden unit 个数
LR = 0.001

# build RNN model
model = Sequential()

# RNN cell
model.add(LSTM(
	batch_input_shape = (BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
  output_dim = CELl_SIZE,
  return_sequences = True,  # False 只输出最后的output True 每一步都输出
  stateful=True,  # batch和batch是否有联系
))
# output layer

model.add(TimeDistributed(Dense(OUTPUT_SIZE)))  # 对时间分散的计算，对每个时间点都介绍

# optimizer
adam = Adam(LR)
model.compilt(optimizer=adam,
             loss = 'mse'
             )
# training
for step in range(40001):
  # data shape = (batch_num, steps, inputs/outputs)
  cost = model.train_on_batch(X_batch, Y_batch)
	pred = model.predict(X_batch, BATCH_SIZE)
```

### keras auto encoder

- 压缩，解压，中间层学到数据精髓。然后再训练学习这些中间层，就可以。

```python
from keras.dataset import mnist
from keras.models import Model
from keras.layers import Dense, Input

# in order to plot in a 2D figure
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape(784,))

# encoder layer
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim,actiavation='relu')(encoded)  # 压缩成2个

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)  # 还原回来了

# construct the autoencoder model
autoencoder = Model(input=input_img, output=decoded)

# construct the encoder model for plotting
encoder = Model(input=input_img, output=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoender.fit(x_train,x_train,nb_epoch=20,,batch_size=256,shuffle=True)

encoded_imgs = encoder.predict(x_test)
```

#### save model load model

```python
from keras.model import load_model
model.save('my_model.h5')  # HDF5
del model
model = load_model('my_model.h5')

# 保存权重
model.save_weight('my_model_weight.h5')
model.load_weight('my_model_weight.h5')

# 保存structure
from keras.model import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)
```

