Brief description

The system comprises the ensemble of six folds of two deep models. It is written with Pytorch v1.0
and requires some other common libraries such as:
- torchvision
- numpy
- scikit-image
- librosa

It is recommended using Mac OS or Linux environment, although it should work on Windows too.
GPU is required for training to get reasonable training time.

Both deep models are based on ResNet with skip connections from bottom layers. Each model
has three instances that were trained on three cross-validation subsets, where 1/3 of training
data was left out for validation. Thus, there are 6 instances of models in total.

Data pre-processing

The data should be accessible at ./data folder. All pre-processing is done during training,
so it is not a separate step/script.

The raw wav data is volume-normalised and converted to a mel-spectrogram with 128 bands.
The spectrogram is further cut to chunks of 128 points long (time-wise) that gives
pseudo-images 128x128 points.

To prevent overfiting the training data is heavily augmented. This includes:
- random change in volume +-30%;
- adding Gaussian noise to spectrogram;
- stretching the time by scaling spectrogram along time axis;
- shifting the pitch by shifting the spectrogram along frequency axis;
- adding random background sounds (other that speech).

The background sounds are taken form an open-source library "ESC-50"
https://github.com/karoldvl/ESC-50

Due to lack of time, the effect of the background sounds has not been studied properly,
but we assume that it improves generalisation on un-seen speech data

====================================================================================
Steps to reproduce the results

1. Download audio files from the ESC-50 library and put to ./background/audio folder
2. Run the background sounds converter that creates a numpy file with re-sampled audio
3. Run the model training script separately for 2 models, 3 folds each:
python3 a_train.py -b3 -n0 -f0
python3 a_train.py -b3 -n0 -f1
python3 a_train.py -b3 -n0 -f2
python3 a_train.py -b3 -n1 -f0
python3 a_train.py -b3 -n1 -f1
python3 a_train.py -b3 -n1 -f2

The training need to be monitored with tensorboard at the directory ./logs

4. The best model snapshot files (from ./save) need to be chosen and their location need to be
added to the a_submit.py script
Then run the script:
python3 a_submit.py

This will produce a submission file "answer.tsv"

We also included 6 pre-trained models that are referred in a_submit.py, so the script can be run
without steps 1-3 to reproduce the results from the leaderboard
