# AlexNet을 이용해 개와 고양이 분류하기

> [이미지 Classification 문제와 딥러닝: AlexNet으로 개vs고양이 분류하기](https://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html)

# 데이터셋


```python
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import tensorflow
import tensorflow as tf

# 데이터셋을 메모리에 로드하고, 학습 및 예측 과정에서 이들을 미니배치 단위로 추출하는 함수
def read_asirra_subset(subset_dir, one_hot=True, sample_size=None):
    # 학습+검증 데이터셋을 읽어들임
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)
    
    if sample_size is not None and sample_size < set_size:
        filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
        set_size = sample_size
    else:
        np.random.shuffle(filename_list)
        
    # 데이터 array들을 메모리 공간에 미리 할당함
    X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32) # 각 이미지들을 256x256 크기로 일괄적으로 Resize
    y_set = np.empty((set_size), dtype=np.uint8)
    
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print("Reading subset data: {}/{}...".format(i, set_size), end='\r')
        
        label = filename.split(".")[0]
        if label == "cat":
            y = 0
        else:
            y = 1
        
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)
        img = resize(img, (256, 256), mode='constant').astype(np.float32)
        
        X_set[i] = img
        y_set[i] = y
        
    if one_hot:
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_set = y_set_oh
        
    print("\nDone")
    
    return X_set, y_set
```


```python
def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflection from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        # Randomly crop patch
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[y:y+crop_l, x:x+crop_l]    # (h, w, C)

        # Randomly reflect patch horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (N, h, w, C)


def corner_center_crop_reflect(images, crop_l):
    """
    Perform 4 corners and center cropping and reflection from images,
    resulting in 10x augmented patches.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, 10, h, w, C).
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        aug_image_orig = []
        # Crop image in 4 corners
        aug_image_orig.append(image[:crop_l, :crop_l])
        aug_image_orig.append(image[:crop_l, -crop_l:])
        aug_image_orig.append(image[-crop_l:, :crop_l])
        aug_image_orig.append(image[-crop_l:, -crop_l:])
        # Crop image in the center
        aug_image_orig.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                                    W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
        aug_image_orig = np.stack(aug_image_orig)    # (5, h, w, C)

        # Flip augmented images and add it
        aug_image_flipped = aug_image_orig[:, :, ::-1]    # (5, h, w, C)
        aug_image = np.concatenate((aug_image_orig, aug_image_flipped), axis=0)    # (10, h, w, C)
        augmented_images.append(aug_image)
    return np.stack(augmented_images)    # shape: (N, 10, h, w, C)


def center_crop(images, crop_l):
    """
    Perform center cropping of images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param crop_l: int, a side length of crop region.
    :return: np.ndarray, shape: (N, h, w, C).
    """
    H, W = images.shape[1:3]
    cropped_images = []
    for image in images:    # image.shape: (H, W, C)
        # Crop image in the center
        cropped_images.append(image[H//2-(crop_l//2):H//2+(crop_l-crop_l//2),
                              W//2-(crop_l//2):W//2+(crop_l-crop_l//2)])
    return np.stack(cropped_images)
```


```python
class DataSet(object):
    def __init__(self, images, labels=None):
        
        '''
        새로운 dataset 객체를 생성함
        '''
        if labels is not None:
            assert images.shape[0] == labels.shape[0], ('Number of examples mimatch, between images and labels')
            
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._indices = np.arange(self._num_examples, dtype=np.uint)
        self._reset()
        
    def _reset(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    # 주어진 batch_size 크기의 mini batch를 현재 데이터셋으로부터 추출하여 반환
    '''
    <AlexNet 논문에서의 학습 단계와 테스트 단계의 데이터 증강(data augmentaion) 방법>
    
    [1] 학습 단계
    
        원본 256×256 크기의 이미지로부터 227×227 크기의 패치(patch)를 랜덤한 위치에서 추출하고, 
        50% 확률로 해당 패치에 대한 수평 방향으로의 대칭 변환(horizontal reflection)을 수행하여, 이미지 하나 당 하나의 패치를 반환함
    
    [2] 테스트 단계
    
        원본 256×256 크기 이미지에서의 좌측 상단, 우측 상단, 좌측 하단, 우측 하단, 중심 위치 각각으로부터 
        총 5개의 227×227 패치를 추출하고, 이들 각각에 대해 수평 방향 대칭 변환을 수행하여 얻은 5개의 패치를 추가하여, 
        이미지 하나 당 총 10개의 패치를 반환함
    
    <AlexNet 논문과 다른 점>
    
    본래 AlexNet 논문에서는 추출되는 패치의 크기가 224×224라고 명시되어 있으나, 본 구현체에서는 227×227로 설정. 
    실제로 온라인 상의 많은 AlexNet 구현체에서 227×227 크기를 채택하고 있으며, 이렇게 해야만 올바른 형태로 구현이 가능.
    
    또, AlexNet 논문에서는 여기에 PCA에 기반한 색상 증강(color augmentation)을 추가로 수행하였는데, 
    본 구현체에서는 구현의 단순화를 위해 이를 반영하지 않음.
    '''
    def next_batch(self, batch_size, shuffle=True, augment=True, is_train=True, fake_data=False):
        if fake_data:
            fake_batch_images = np.random.random(size=(batch_size, 227, 227, 3))
            fake_batch_labels = np.zeros((batch_size, 2), dtype=np.uint8)
            fake_batch_labels[np.arange(batch_size), np.random.randint(2, size=batch_size)] = 1
            
            return fake_batch_images, fake_batch_labels
        
        start_index = self._index_in_epoch

        # 맨 첫 번째 epoch에서는 전체 데이터셋을 랜덤하게 섞음
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            np.random.shuffle(self._indices)

        # 현재의 인덱스가 전체 이미지 수를 넘어간 경우, 다음 epoch를 진행함
        if start_index + batch_size > set._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start_index
            indices_rest_part = selt._indices[start_index : self._num_examples]

            # 하나의 epoch이 끝나면, 전체 데이터셋을 섞음
            if shuffle:
                np.random.shuffle(self._indices)

            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index : end_index]

            images_rest_part = self.images[indices_rest_part]
            images_new_part = self.images[indices_new_part]
            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)

            if self.labels is not None:
                labels_rest_part = self.labels[indices_rest_part]
                labels_new_part = self.labels[indices_new_part]
                batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)

            else:
                batch_labels = None

        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            indices = self._indices[start_index : end_index]
            batch_images = self.images[indices]

            if self.labels is not None:
                batch_labels = self.labels[indices]
            else:
                batch_labels = None

        if augment and is_train:
            # 학습 상황에서의 데이터 증강을 수행함
            batch_images = random_crop_reflect(batch_images, 227)
        elif augment and not is_train:
            # 예측 상황에서의 데이터 증강을 수행함
            batch_images = corner_center_crop_reflect(batch_images, 227)
        else:
            # 데이터 증강을 수행하지 않고, 단순히 이미지 중심 위치에서만 추출된 패치를 사용함
            batch_images = center_crop(batch_images, 227)

        return batch_images, batch_labels
```

# 성능 평가: 정확도


```python
from abc import abstractmethod, abstractproperty
from sklearn.metrics import accuracy_score

# evaluator를 서술하는 베이스 클래스로, 성능 평가 척도에 따라 '최저' 성능 점수와 '점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지' 등이 다르기 때문에,
# 이들을 명시하는 부분이 각각 worst_score와 mode이다.
class Evaluator(object):
    """성능 평가를 위한 evaluator의 베이스 클래스."""

    @abstractproperty
    def worst_score(self):
        """
        최저 성능 점수.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나.
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        실제로 사용할 성능 평가 지표.
        해당 함수를 추후 구현해야 함.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        해당 함수를 추후 구현해야 함.
        :param curr: float, 평가 대상이 되는 현재 성능 점수.
        :param best: float, 현재까지의 최고 성능 점수.
        :return bool.
        """
        pass

# 정확도를 평가 척도로 삼는 evaluator로, evluator 클래스를 구현한 것이다.
class AccuracyEvaluator(Evaluator):
    """정확도를 평가 척도로 사용하는 evaluator 클래스."""

    @property
    def worst_score(self):
        """최저 성능 점수."""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부."""
        return 'max'

    def score(self, y_true, y_pred):
        """정확도에 기반한 성능 평가 점수."""
        return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    
    # 두 성능 간의 단순 비교를 수행하는 것이 아니라, 상대적 문턱값(relative threshold)을 사용하여 현재 평가 성능이 최고 평가 성능보다 지정한 비율 이상으로 높은 경우에 한하여 True를 반환하도록 한다.    
    def is_better(self, curr, best, **kwargs):
        """
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다
        우수한지 여부를 반환하는 함수.
        :param kwargs: dict, 추가 인자.
            - score_threshold: float, 새로운 최적값 결정을 위한 상대적 문턱값으로,
                               유의미한 차이가 발생했을 경우만을 반영하기 위함.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps
```

# 러닝 모델: AlexNet


```python
# 러닝 모델을 사후적으로 수정하거나 혹은 새로운 구조의 러닝 모델을 추가하는 상황에서의 편의를 고려하여, 컨볼루션 신경망에서 주로 사용하는 층들을 생성하는 함수를 미리 정의해 놓고,
# 일반적인 컨볼루션 신경망 모델을 표현하는 베이스 클래스를 먼저 정의한 뒤 이를 AlexNet의 클래스가 상속받는 형태로 구현함.
def weight_variable(shape, stddev=0.01):
    """
    새로운 가중치 변수를 주어진 shape에 맞게 선언하고,
    Normal(0.0, stddev^2)의 정규분포로부터의 샘플링을 통해 초기화함.
    :param shape: list(int).
    :param stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
    :return weights: tf.Variable.
    """
    weights = tf.get_variable('weights', shape, tf.float32,
                              tf.random_normal_initializer(mean=0.0, stddev=stddev))
    return weights


def bias_variable(shape, value=1.0):
    """
    새로운 바이어스 변수를 주어진 shape에 맞게 선언하고, 
    주어진 상수값으로 추기화함.
    :param shape: list(int).
    :param value: float, 바이어스의 초기화 값.
    :return biases: tf.Variable.
    """
    biases = tf.get_variable('biases', shape, tf.float32,
                             tf.constant_initializer(value=value))
    return biases


def conv2d(x, W, stride, padding='SAME'):
    """
    주어진 입력값과 필터 가중치 간의 2D 컨볼루션을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param W: tf.Tensor, shape: (fh, fw, ic, oc).
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, side_l, stride, padding='SAME'):
    """
    주어진 입력값에 대해 최댓값 풀링(max pooling)을 수행함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 풀링 윈도우의 한 변의 길이.
    :param stride: int, 풀링 윈도우의 각 방향으로의 이동 간격. 
    :param padding: str, 'SAME' 또는 'VALID',
                         풀링 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :return: tf.Tensor.
    """
    return tf.nn.max_pool(x, ksize=[1, side_l, side_l, 1],
                          strides=[1, stride, stride, 1], padding=padding)


def conv_layer(x, side_l, stride, out_depth, padding='SAME', **kwargs):
    """
    새로운 컨볼루션 층을 추가함.
    :param x: tf.Tensor, shape: (N, H, W, C).
    :param side_l: int, 필터의 한 변의 길이.
    :param stride: int, 필터의 각 방향으로의 이동 간격.
    :param out_depth: int, 입력값에 적용할 필터의 총 개수.
    :param padding: str, 'SAME' 또는 'VALID',
                         컨볼루션 연산 시 입력값에 대해 적용할 패딩 알고리즘.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함.
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_depth = int(x.get_shape()[-1])

    filters = weight_variable([side_l, side_l, in_depth, out_depth], stddev=weights_stddev)
    biases = bias_variable([out_depth], value=biases_value)
    return conv2d(x, filters, stride, padding=padding) + biases


def fc_layer(x, out_dim, **kwargs):
    """
    새로운 완전 연결 층을 추가함.
    :param x: tf.Tensor, shape: (N, D).
    :param out_dim: int, 출력 벡터의 차원수.
    :param kwargs: dict, 추가 인자, 가중치/바이어스 초기화를 위한 하이퍼파라미터들을 포함함. 
        - weight_stddev: float, 샘플링 대상이 되는 정규분포의 표준편차 값.
        - biases_value: float, 바이어스의 초기화 값.
    :return: tf.Tensor.
    """
    weights_stddev = kwargs.pop('weights_stddev', 0.01)
    biases_value = kwargs.pop('biases_value', 0.1)
    in_dim = int(x.get_shape()[-1])

    weights = weight_variable([in_dim, out_dim], stddev=weights_stddev)
    biases = bias_variable([out_dim], value=biases_value)
    return tf.matmul(x, weights) + biases
```


```python
import time
from abc import abstractmethod
import numpy as np

# Convolution 신경망 모델
class ConvNet(object):
    """컨볼루션 신경망 모델의 베이스 클래스."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        모델 생성자.
        :param input_shape: tuple, shape (H, W, C) 및 값 범위 [0.0, 1.0]의 입력값.
        :param num_classes: int, 총 클래스 개수.
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape)
        self.y = tf.placeholder(tf.float32, [None] + [num_classes])
        self.is_train = tf.placeholder(tf.bool)

        # 모델과 손실 함수 정의
        self.d = self._build_model(**kwargs)
#         self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        모델 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        해당 함수를 추후 구현해야 함. 
        """
        pass

    # dataset을 입력받아 이에 대한 모델의 예측 결과를 반환한다.
    # 이때, 테스트 단계에서 데이터 증강 방법을 채택할 경우(augment_pred=True), 하나의 이미지 당 총 10개의 패치를 얻으며, 이들 각각에 대한
    # 예측 결과를 계산하고 이들의 평균을 계산하는 방식으로 최종적으로 예측을 수행한다.
    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        주어진 데이터셋에 대한 예측을 수행함.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, 예측 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 예측을 위한 추가 인자.
            - batch_size: int, 각 반복 회차에서의 미니배치 크기.
            - augment_pred: bool, 예측 과정에서 데이터 증강을 수행할지 여부.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # 예측 루프를 시작함
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # 예측 과정에서 데이터 증강을 수행할 경우,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # 10종류의 patch 각각에 대하여 예측 결과를 산출하고,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # 이들 10개 예측 결과의 평균을 산출함
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # 예측 결과를 단순 산출함
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total evaluation time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred
```


```python
class AlexNet(ConvNet):
    """AlexNet 클래스."""

    def _build_model(self, **kwargs):
        """
        모델 생성.
        :param kwargs: dict, AlexNet 생성을 위한 추가 인자.
            - image_mean: np.ndarray, 평균 이미지: 이미지들의 각 입력 채널별 평균값, shape: (C,).
            - dropout_prob: float, 완전 연결 층에서 각 유닛별 드롭아웃 수행 확률.
        :return d: dict, 각 층에서의 출력값들을 포함함.
        """
        d = dict()    # 각 중간층에서의 출력값을 포함하는 dict.
        X_mean = kwargs.pop('image_mean', 0.0)
        dropout_prob = kwargs.pop('dropout_prob', 0.0)
        num_classes = int(self.y.get_shape()[-1])

        # Dropout을 적용할 층들에서의 각 유닛별 '유지' 확률
        keep_prob = tf.cond(self.is_train,
                            lambda: 1. - dropout_prob,
                            lambda: 1.)

        # input
        X_input = self.X - X_mean    # 기존 입력값으로부터 평균 이미지를 뺌

        # conv1 - relu1 - pool1
        with tf.variable_scope('conv1'):
            d['conv1'] = conv_layer(X_input, 11, 4, 96, padding='VALID',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv1.shape', d['conv1'].get_shape().as_list())
        d['relu1'] = tf.nn.relu(d['conv1'])
        # (227, 227, 3) --> (55, 55, 96)
        d['pool1'] = max_pool(d['relu1'], 3, 2, padding='VALID')
        # (55, 55, 96) --> (27, 27, 96)
        print('pool1.shape', d['pool1'].get_shape().as_list())

        # conv2 - relu2 - pool2
        with tf.variable_scope('conv2'):
            d['conv2'] = conv_layer(d['pool1'], 5, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv2.shape', d['conv2'].get_shape().as_list())
        d['relu2'] = tf.nn.relu(d['conv2'])
        # (27, 27, 96) --> (27, 27, 256)
        d['pool2'] = max_pool(d['relu2'], 3, 2, padding='VALID')
        # (27, 27, 256) --> (13, 13, 256)
        print('pool2.shape', d['pool2'].get_shape().as_list())

        # conv3 - relu3
        with tf.variable_scope('conv3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.0)
            print('conv3.shape', d['conv3'].get_shape().as_list())
        d['relu3'] = tf.nn.relu(d['conv3'])
        # (13, 13, 256) --> (13, 13, 384)

        # conv4 - relu4
        with tf.variable_scope('conv4'):
            d['conv4'] = conv_layer(d['relu3'], 3, 1, 384, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv4.shape', d['conv4'].get_shape().as_list())
        d['relu4'] = tf.nn.relu(d['conv4'])
        # (13, 13, 384) --> (13, 13, 384)

        # conv5 - relu5 - pool5
        with tf.variable_scope('conv5'):
            d['conv5'] = conv_layer(d['relu4'], 3, 1, 256, padding='SAME',
                                    weights_stddev=0.01, biases_value=0.1)
            print('conv5.shape', d['conv5'].get_shape().as_list())
        d['relu5'] = tf.nn.relu(d['conv5'])
        # (13, 13, 384) --> (13, 13, 256)
        d['pool5'] = max_pool(d['relu5'], 3, 2, padding='VALID')
        # (13, 13, 256) --> (6, 6, 256)
        print('pool5.shape', d['pool5'].get_shape().as_list())

        # 전체 feature maps를 flatten하여 벡터화
        f_dim = int(np.prod(d['pool5'].get_shape()[1:]))
        f_emb = tf.reshape(d['pool5'], [-1, f_dim])
        # (6, 6, 256) --> (9216)

        # fc6
        with tf.variable_scope('fc6'):
            d['fc6'] = fc_layer(f_emb, 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu6'] = tf.nn.relu(d['fc6'])
        d['drop6'] = tf.nn.dropout(d['relu6'], keep_prob)
        # (9216) --> (4096)
        print('drop6.shape', d['drop6'].get_shape().as_list())

        # fc7
        with tf.variable_scope('fc7'):
            d['fc7'] = fc_layer(d['drop6'], 4096,
                                weights_stddev=0.005, biases_value=0.1)
        d['relu7'] = tf.nn.relu(d['fc7'])
        d['drop7'] = tf.nn.dropout(d['relu7'], keep_prob)
        # (4096) --> (4096)
        print('drop7.shape', d['drop7'].get_shape().as_list())

        # fc8
        with tf.variable_scope('fc8'):
            d['logits'] = fc_layer(d['relu7'], num_classes, weights_stddev=0.01, biases_value=0.0)
        # (4096) --> (num_classes)

        # softmax
        d['pred'] = tf.nn.softmax(d['logits'])

        return d

    def _build_loss(self, **kwargs):
        """
        모델 학습을 위한 손실 함수 생성.
        :param kwargs: dict, 정규화 항을 위한 추가 인자.
            - weight_decay: float, L2 정규화 계수.
        :return tf.Tensor.
        """
        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = tf.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # 소프트맥스 교차 엔트로피 손실 함수
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y) #, logits=self.logits)
        softmax_loss = tf.reduce_mean(softmax_losses)

        return softmax_loss + weight_decay*l2_reg_loss
```

# 러닝 알고리즘: SGD + Momentum


```python
import os
import time
from abc import abstractmethod

# AlexNet 논문에 따라 모멘텀(momentum)을 적용한 SGD를 채택
class Optimizer(object):
    """경사 하강 러닝 알고리즘 기반 optimizer의 베이스 클래스."""

    def __init__(self, model, train_set, evaluator, val_set=None, **kwargs):
        """
        optimizer 생성자.
        :param model: ConvNet, 학습할 모델.
        :param train_set: DataSet, 학습에 사용할 학습 데이터셋.
        :param evaluator: Evaluator, 학습 수행 과정에서 성능 평가에 사용할 evaluator.
        :param val_set: DataSet, 검증 데이터셋, 주어지지 않은 경우 None으로 남겨둘 수 있음.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
            - batch_size: int, 각 반복 회차에서의 미니배치 크기.
            - num_epochs: int, 총 epoch 수.
            - init_learning_rate: float, 학습률 초깃값.
        """
        self.model = model
        self.train_set = train_set
        self.evaluator = evaluator
        self.val_set = val_set

        # 학습 관련 하이퍼파라미터
        self.batch_size = kwargs.pop('batch_size', 256)
        self.num_epochs = kwargs.pop('num_epochs', 320)
        self.init_learning_rate = kwargs.pop('init_learning_rate', 0.01)

        self.learning_rate_placeholder = tf.placeholder(tf.float32)    # 현 학습률 값의 Placeholder
        self.optimize = self._optimize_op()

        self._reset()

    def _reset(self):
        """일부 변수를 재설정."""
        self.curr_epoch = 1
        self.num_bad_epochs = 0    # 'bad epochs' 수: 성능 향상이 연속적으로 이루어지지 않은 epochs 수.
        self.best_score = self.evaluator.worst_score    # 최저 성능 점수로, 현 최고 점수를 초기화함.
        self.curr_learning_rate = self.init_learning_rate    # 현 학습률 값

    @abstractmethod
    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.Optimizer.minimize Op.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    @abstractmethod
    def _update_learning_rate(self, **kwargs):
        """
        고유의 학습률 스케줄링 방법에 따라, (필요한 경우) 매 epoch마다 현 학습률 값을 업데이트함.
        해당 함수를 추후 구현해야 하며, 외부에서 임의로 호출할 수 없음.
        """
        pass

    def _step(self, sess, **kwargs):
        """
        경사 하강 업데이트를 1회 수행하며, 관련된 값을 반환함.
        해당 함수를 외부에서 임의로 호출할 수 없음.
        :param sess: tf.Session.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
            - augment_train: bool, 학습 과정에서 데이터 증강을 수행할지 여부.
        :return loss: float, 1회 반복 회차 결과 손실 함숫값.
                y_true: np.ndarray, 학습 데이터셋의 실제 레이블.
                y_pred: np.ndarray, 모델이 반환한 예측 레이블.
        """
        augment_train = kwargs.pop('augment_train', True)

        # 미니배치 하나를 추출함
        X, y_true = self.train_set.next_batch(self.batch_size, shuffle=True,
                                              augment=augment_train, is_train=True)

        # 손실 함숫값을 계산하고, 모델 업데이트를 수행함
        _, loss, y_pred = \
            sess.run([self.optimize, self.model.loss, self.model.pred],
                     feed_dict={self.model.X: X, self.model.y: y_true,
                                self.model.is_train: True,
                                self.learning_rate_placeholder: self.curr_learning_rate})

        return loss, y_true, y_pred

    def train(self, sess, save_dir='/tmp', details=False, verbose=True, **kwargs):
        """
        optimizer를 실행하고, 모델을 학습함.
        :param sess: tf.Session.
        :param save_dir: str, 학습된 모델의 파라미터들을 저장할 디렉터리 경로.
        :param details: bool, 학습 결과 관련 구체적인 정보를, 학습 종료 후 반환할지 여부.
        :param verbose: bool, 학습 과정에서 구체적인 정보를 출력할지 여부.
        :param kwargs: dict, 학습 관련 하이퍼파라미터로 구성된 추가 인자.
        :return train_results: dict, 구체적인 학습 결과를 담은 dict.
        """
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())    # 전체 파라미터들을 초기화함

        train_results = dict()    # 학습 (및 검증) 결과 관련 정보를 포함하는 dict.
        train_size = self.train_set.num_examples
        num_steps_per_epoch = train_size // self.batch_size
        num_steps = self.num_epochs * num_steps_per_epoch

        if verbose:
            print('Running training loop...')
            print('Number of training iterations: {}'.format(num_steps))

        step_losses, step_scores, eval_scores = [], [], []
        start_time = time.time()

        # 학습 루프를 실행함
        for i in range(num_steps):
            # 미니배치 하나로부터 경사 하강 업데이트를 1회 수행함
            step_loss, step_y_true, step_y_pred = self._step(sess, **kwargs)
            step_losses.append(step_loss)

            # 매 epoch의 말미에서, 성능 평가를 수행함
            if (i+1) % num_steps_per_epoch == 0:
                # 학습 데이터셋으로부터 추출한 현재의 미니배치에 대하여 모델의 예측 성능을 평가함
                step_score = self.evaluator.score(step_y_true, step_y_pred)
                step_scores.append(step_score)

                # 검증 데이터셋이 처음부터 주어진 경우, 이를 사용하여 모델 성능을 평가함
                if self.val_set is not None:
                    # 검증 데이터셋을 사용하여 모델 성능을 평가함
                    eval_y_pred = self.model.predict(sess, self.val_set, verbose=False, **kwargs)
                    eval_score = self.evaluator.score(self.val_set.labels, eval_y_pred)
                    eval_scores.append(eval_score)

                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {:.6f} |Train score: {:.6f} |Eval score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, eval_score, self.curr_learning_rate))
                        # 중간 결과를 플롯팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=eval_scores,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = eval_score

                # 그렇지 않은 경우, 단순히 미니배치에 대한 결과를 사용하여 모델 성능을 평가함
                else:
                    if verbose:
                        # 중간 결과를 출력함
                        print('[epoch {}]\tloss: {} |Train score: {:.6f} |lr: {:.6f}'\
                              .format(self.curr_epoch, step_loss, step_score, self.curr_learning_rate))
                        # 중간 결과를 플릇팅함
                        plot_learning_curve(-1, step_losses, step_scores, eval_scores=None,
                                            mode=self.evaluator.mode, img_dir=save_dir)
                    curr_score = step_score

                # 현재의 성능 점수의 현재까지의 최고 성능 점수를 비교하고, 
                # 최고 성능 점수가 갱신된 경우 해당 성능을 발휘한 모델의 파라미터들을 저장함
                if self.evaluator.is_better(curr_score, self.best_score, **kwargs):
                    self.best_score = curr_score
                    self.num_bad_epochs = 0
                    saver.save(sess, os.path.join(save_dir, 'model.ckpt'))    # 현재 모델의 파라미터들을 저장함
                else:
                    self.num_bad_epochs += 1

                self._update_learning_rate(**kwargs)
                self.curr_epoch += 1

        if verbose:
            print('Total training time(sec): {}'.format(time.time() - start_time))
            print('Best {} score: {}'.format('evaluation' if eval else 'training',
                                             self.best_score))
        print('Done.')

        if details:
            # 학습 결과를 dict에 저장함
            train_results['step_losses'] = step_losses    # (num_iterations)
            train_results['step_scores'] = step_scores    # (num_epochs)
            if self.val_set is not None:
                train_results['eval_scores'] = eval_scores    # (num_epochs)

            return train_results
```


```python
class MomentumOptimizer(Optimizer):
    """모멘텀 알고리즘을 포함한 경사 하강 optimizer 클래스."""

    def _optimize_op(self, **kwargs):
        """
        경사 하강 업데이트를 위한 tf.train.MomentumOptimizer.minimize Op.
        :param kwargs: dict, optimizer의 추가 인자.
            - momentum: float, 모멘텀 계수.
        :return tf.Operation.
        """
        momentum = kwargs.pop('momentum', 0.9)

        update_vars = tf.trainable_variables()
        return tf.train.MomentumOptimizer(self.learning_rate_placeholder, momentum, use_nesterov=False)\
                .minimize(self.model.loss, var_list=update_vars)

    def _update_learning_rate(self, **kwargs):
        """
        성능 평가 점수 상에 개선이 없을 때, 현 학습률 값을 업데이트함.
        :param kwargs: dict, 학습률 스케줄링을 위한 추가 인자.
            - learning_rate_patience: int, 성능 향상이 연속적으로 이루어지지 않은 epochs 수가 
                                      해당 값을 초과할 경우, 학습률 값을 감소시킴.
            - learning_rate_decay: float, 학습률 업데이트 비율.
            - eps: float, 업데이트된 학습률 값과 기존 학습률 값 간의 차이가 해당 값보다 작을 경우,
                          학습률 업데이트를 취소함.
        """
        learning_rate_patience = kwargs.pop('learning_rate_patience', 10)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.1)
        eps = kwargs.pop('eps', 1e-8)

        if self.num_bad_epochs > learning_rate_patience:
            new_learning_rate = self.curr_learning_rate * learning_rate_decay
            # 새 학습률 값과 기존 학습률 값 간의 차이가 eps보다 큰 경우에 한해서만 업데이트를 수행함
            if self.curr_learning_rate - new_learning_rate > eps:
                self.curr_learning_rate = new_learning_rate
            self.num_bad_epochs = 0
```

# 학습 수행 및 테스트 결과

## 훈련


```python
""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('/root/share/', 'asirra')    # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = read_asirra_subset(trainval_dir, one_hot=True)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.2)    # FIXME
val_set = DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = DataSet(X_trainval[val_size:], y_trainval[val_size:])

# 중간 점검
print('Training set stats:')
print(train_set.images.shape)
print(train_set.images.min(), train_set.images.max())
print((train_set.labels[:, 1] == 0).sum(), (train_set.labels[:, 1] == 1).sum())
print('Validation set stats:')
print(val_set.images.shape)
print(val_set.images.min(), val_set.images.max())
print((val_set.labels[:, 1] == 0).sum(), (val_set.labels[:, 1] == 1).sum())


""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = train_set.images.mean(axis=(0, 1, 2))    # 평균 이미지
np.save('/root/share/asirra/result/asirra_mean.npy', image_mean)    # 평균 이미지를 저장
hp_d['image_mean'] = image_mean

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['num_epochs'] = 300

hp_d['augment_train'] = True
hp_d['augment_pred'] = True

hp_d['init_learning_rate'] = 0.01
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 30
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8

# FIXME: 정규화 관련 하이퍼파라미터
hp_d['weight_decay'] = 0.0005
hp_d['dropout_prob'] = 0.5

# FIXME: 성능 평가 관련 하이퍼파라미터
hp_d['score_threshold'] = 1e-4


""" 3. Graph 생성, session 초기화 및 학습 시작 """
# 초기화
graph = tf.get_default_graph()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)
sess = tf.Session(graph=graph, config=config)
    
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)
```

## 테스트


```python
""" 1. 원본 데이터셋을 메모리에 로드함 """
root_dir = os.path.join('/root/share/', 'Datasets', 'asirra')    # FIXME
test_dir = os.path.join(root_dir, 'test')

# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_asirra_subset(test_dir, one_hot=True)
test_set = dataset.DataSet(X_test, y_test)

# 중간 점검
print('Test set stats:')
print(test_set.images.shape)
print(test_set.images.min(), test_set.images.max())
print((test_set.labels[:, 1] == 0).sum(), (test_set.labels[:, 1] == 1).sum())


""" 2. 테스트를 위한 하이퍼파라미터 설정 """
hp_d = dict()
image_mean = np.load('/root/share/asirra/result/asirra_mean.npy')    # 평균 이미지를 로드
hp_d['image_mean'] = image_mean

# FIXME: 테스트 관련 하이퍼파라미터
hp_d['batch_size'] = 256
hp_d['augment_pred'] = True


""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화 
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([227, 227, 3], 2, **hp_d)
evaluator = Evaluator()
saver = tf.train.Saver()

sess = tf.Session(graph=graph, config=config)
saver.restore(sess, '/root/share/asirra/result/model.ckpt')    # 학습된 파라미터 로드 및 복원
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test accuracy: {}'.format(test_score))
```
