# EEG-user-identifications
1D-convolutional LSTM based User Identification using EEG biometrics

This repository is for paper "EEG-based user identification system using 1D-convolutional long short-term memory neural networks". PhysioNet EEG Motor Movement/Imagery Dataset was used for the training of the network. It can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/

EEG data was resampled using a sliding window in matlab data preprocessing and stored in mat files, which can be read in python with h5py library. In the paper, a 3-fold cross-validation was performed, therefore, 12 out of 14 data collection sessions were evenly divided into 3 groups. During cross-validation, 2 out of 3 groups were used for training and the other one was used only for testing.

Please refer to the paper for more details:

@article{
  title={EEG-based user identification system using 1D-convolutional long short-term memory neural networks},
  author={Sun, Yingnan and Lo, Frank P-W and Lo, Benny},
  journal={Expert Systems with Applications},
  volume={125},
  pages={259--267},
  year={2019},
  publisher={Elsevier}
}

If you have any questions, please feel free to email ys4315@ic.ac.uk

