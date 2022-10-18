# 학습방법 선택

<br/><br/>

## 지도학습

<br/><br/>

## 비지도학습

<br/><br/>

## 강화학습

<br/><br/><br/><br/>



# 알고리즘 선택

<br/><br/>

## 일반 ML

<br/><br/>

## 신경망 

<br/>

### 개요

<br/>

#### Dense(units=?), ConvND(filters=?) 값은 충분히 클 수록 좋다.
* 다중 분류(예: classes=40)에서 모델의 중간층은 최소 다중 분류항의 갯 수 보다 만아야 함(예: 최소 40*2=80개 이상)
* 샘플 개수가 적을 경우(예 1000개 미만) 은닉층을 2개 정도만 한다.
  * 샘플 개수가 적을 수록 `과대적합`이 쉽게 이러나며, `작은 모델`이 과대적합을 피하는 방법이다.

<br/>

#### 상황에 따른 손실함수 선택
* 2진분류: `binary_crossentropy`
* n진분류: 
  * [[1 0 0], [0 1 0]..] one-hot-encoding: categorical_crossentropy
  * [1 2 3 4..] 정수형 분류: sparse_categorical_crossentropy
* 회귀: `평균제곱오차(MSE), mean_squared_error`
* 시계열예측: `CTC(Connection Temporal Classification), 좀더 확인 필요(keras에 구현되어 있지 않음)`
* 회귀: activation func 없이 `MSE` 함수 사용, 평가지표는 `MAE` 사용(accuracy 아님)

<br/>

### DNN(Deep Neural Network)
<br/>

### CNN(Convolutional Neural Network)
<br/>

### RNN(Recurrent Neural Networks)
<br/>

#### LSTM(Long Short-Term Memory)

<br/>

### GAN(Generative Adversarial Network)
<br/>