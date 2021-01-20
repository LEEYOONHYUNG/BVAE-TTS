from text import symbols


################################
# Experiment Parameters        #
################################
seed=1234
output_directory = 'training_log'
iters_per_validation=1000
iters_per_checkpoint=10000

data_path = '../Dataset/LJSpeech-1.1/preprocessed'
training_files='filelists/ljs_audio_text_train_filelist.txt'
validation_files='filelists/ljs_audio_text_val_filelist.txt'
test_files='filelists/ljs_audio_text_val_filelist.txt'
text_cleaners=['english_cleaners']


################################
# Audio Parameters             #
################################
sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin=0
mel_fmax=8000.0


################################
# Model Parameters             #
################################
n_symbols=len(symbols)
data_type='phone_seq'
n_blocks=4
n_layers=3
kernel_size=5
downsample_ratio=4
symbols_embedding_dim=256
hidden_dim=256
max_db=2
min_db=-12


################################
# Optimization Hyperparameters #
################################
lr=1e-3
lr_warmup_steps=4000
kl_warmup_steps=60000
grad_clip_thresh=1.0
batch_size=128
train_steps = 300000

