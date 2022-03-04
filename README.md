# DNN-2d-FDTD
## Dynamics
<img src="./3DCNNEnDeLSTMSourcePEC/FDTD_vs_DNN.gif" alt="" width="500"/>

<img src="./General 3DCNNEnDeLSTMSourcePEC_1/FDTD_vs_DNN_1.gif" alt="" width="500"/>

<img src="./General 3DCNNEnDeLSTMSourcePEC_2/FDTD_vs_DNN1.gif" alt="" width="500"/>

## Breif Introduction
This project contains multiple subprojects towards using DNN to simulate the behavior of 2D FDTD. Multiple network architecure are tries:
* 3D-CNN + LSTM with Encoder & Decoder
* ConvLSTM with Decoder
* CNN Encoder & Decoder
* LSTM with Encoder & Decoder

## Figures
training epoch70:

<img src="./3DCNNEnDeLSTMSourcePEC/epoch70.png" alt="" width="500"/>

training epoch120:

<img src="./3DCNNEnDeLSTMSourcePEC/epoch120.png" alt="" width="500"/>

training epoch11000:

<img src="./3DCNNEnDeLSTMSourcePEC/epoch11000.png" alt="" width="500"/>

training loss:

<img src="./General 3DCNNEnDeLSTMSourcePEC_1/loss 50 eps 1e-6 lr.png" alt="" width="500"/>

All networks in this are light-weight and can be trained on GTX 1050 Laptop.
