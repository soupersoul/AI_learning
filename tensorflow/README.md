# AI_learning

TensorFlow学习笔记（一）入门

http://blog.csdn.net/WuyZhen_CSDN/article/details/64516733

http://blog.csdn.net/column/details/13300.html?&page=3

jupyter里使用!可以执行命令

激活函数：tf.nn.*
- relu: y = max(0, x)
- softmax: 将向量变成概率分布。
   x = [x1, x2, x3],  y = [e^x1/sum, e^x2/sum, e^x3/sum],   sum = e^x1 + e^x2 + e^x3S
- sigmoid:  1/(1+e^-x)  # 比较适合二分类
- softplus: log(1+e^x), 比relu平滑，不会出现折点,  x> 0 时，约等于ｘ,　ｘ小于０时则很接近于0
selu,  elu,

损失函数：

    分类
    - sparse_categorical_crossentropy
    - categorical_crossentropy
    - binary_crossentropy
    回归
    - mean_squared_error

优化器：
- sgd
- adam

metrics:
- accuracy

API:
keras.callbacks.TensorBoard(logdir)
keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True)

keras.estimator.model_to_estimator(model)   # if data is pd, then change to dict for estimator
tf.estimator.BaselineClassifier # BaselineClassifier： 根据分类在出现在样本中的比例，来进行预测类别，没有什么模型，只是根据比例来随机猜测
tf.estimator.LinearClassifier
tf.estimator.DNNClassifier

keras.applications.ResNet50

keras.layers.Activation
keras.layers.AlphaDropout(rate=0.5) # 一般都用0.5, AlphaDropout优点：1. 均值和方差不变  2.归一化性质也不变（非常重要的优点）
keras.layers.BatchNormalization()
keras.layers.concatenate
keras.layers.Conv2D
keras.layers.SeparableConv2D
keras.layers.Dense
keras.layers.DenseFeatures
keras.layers.Dropout(rate=0.5)
keras.layers.Embedding
keras.layers.Flatten
keras.layers.GlobalAveragePooling1D
keras.layers.GRU
keras.layers.Input
keras.layers.Lambda(lambda x : tf.nn.softplus(x))
keras.layers.Layer
keras.layers.LSTM
keras.layers.MaxPool2D
keras.layers.Bidirectional(
    keras.layers.SimpleRNN(units = 64, return_sequences  =True), # return_sequences=True, 返回的结果是所有的输出；False，返回的结果是最后的输出
)

keras.losses.mean_squared_error
keras.losses.SparseCategoricalCfrossentropy
keras.losses.Reduction.SUM_OVER_BATCH_SIZE

keras.metrics.MeanSquaredError

keras.models.Model
keras.models.Sequential

keras.optimizers.SGD


train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1/.255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest',
)
keras.preprocessing.image.ImageDataGenerator.flow_from_directory(train_dir,
                                                   target_size = (height, width),
                                                   batch_size = batch_size,
                                                   seed = 7,
                                                   shuffle = True,
                                                   class_mode = "categorical")
keras.preprocessing.image.ImageDataGenerator.flow_from_dataframe
keras.preprocessing.sequence.pad_sequences
keras.preprocessing.text.Tokenizer(num_words=None, filters='', split=' ')


keras.wrappers.scikit_learn.KerasRegressor  (keras model to sklearn model)

#归一化 x = (x - u) / std
sklearn.preprocessing.StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal
#随机函数reciprocal的概率分布: f(x) = 1/(x*log(b/a))    a <= x <= b

tf.data.Dataset.list_files
tf.data.Dataset.repeat
tf.data.Dataset.interleave
tf.data.Dataset.from_tensor_slices
tf.data.Dataset.make_one_shot_iterator

tf.data.TextLineDataset(filename).skip(1)   #把csv文件按行转成dataset,skip前n行
tf.data.TFRecordDataset

tf.estimator.RunConfig

tf.feature_column.input_layer
tf.feature_column.indicator_column # to one-hot column
tf.feature_column.numeric_column
tf.feature_column.crossed_column
tf.io.decode_csv
tf.io.FixedLenFeature
tf.io.parse_single_example
tf.io.TFRecordOptions
tf.io.TFRecordWriter
tf.io.VarLenFeature
tf.distribute.MirroredStrategyzaohua

tf.random.categorical
tf.expand_dims 扩展维度
tf.squeeze       降低维度


tf.sparse.to_dense
tf.train.FloatList
tf.train.Features
tf.train.Feature
tf.train.Example
  example.SerializeToString<-->tf.io.parse_single_example
#tfrecord file format
#-> tf.train.Example
#--> tf.train.Features -> { "key": tf.train.Feature }
#---> tf.train.Feature -> tf.train.ByteList/FloatList/Int64List


tf.greater_equal
tf.where
tf.function
    get_concrete_function
tf.autograph.to_code
tf.TensorSpec


  
tf.debugging.set_log_device_placement
tf.config.experimental.set_visible_devices
tf.config.experimental.list_logical_devices
tf.config.experimental.list_physical_devices
tf.config.experimental.set_memory_growth
tf.config.experimental.VirtualDeviceConfiguration
tf.config.set_soft_device_placement

# numpy(np)
np.random.randint
np.c_   # 按行拼接


# 实时监视GPU的使用量
watch -n 0.1 -x nvidia-smi

# TF1.0
tf.layers.dense(x, hidden_unit, activation = tf.nn.relu)
tf.losses.sparse_softmax_cross_entropy
