# Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech (BVAE-TTS)  
## Yoonhyung Lee, Joongbo Shin, Kyomin Jung  
Implementation of ["Bidirectional Variational Inference for Non-Autoregressive Text-to-Speech"](https://openreview.net/forum?id=o3iritJHLfO)  
  

## Training  
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)  
2. Make `preprocessed` folder in LJSpeech directory and do preprocessing of the data using `prepare_data.ipynb`  
3. Set the `data_path` in `hparams.py` to the `preprocessed` folder  
```python
python train.py --gpu=0 --logdir=baseline  
```  


## Pre-trained models  
- [BVAE-TTS](http://milabfile.snu.ac.kr:16000/bvae-tts/bvae_tts_300k.pt)  
- [WaveGlow](http://milabfile.snu.ac.kr:16000/bvae-tts/waveglow_256channels.pt)  


## Audio Samples  
You can hear the audio samples [here](https://leeyoonhyung.github.io/Transformer-TTS/)  


## Notice  
1. Unlike the original paper, I didn't use the encoder-prenet following [espnet](https://github.com/espnet/espnet)  
2. I apply additional ["guided attention loss"](https://arxiv.org/pdf/1710.08969.pdf) to the two heads of the last two layers  


## Reference
1.NVIDIA/tacotron2: https://github.com/NVIDIA/tacotron2  
2.espnet/espnet: https://github.com/espnet/espnet  
3.soobinseo/Transformer-TTS: https://github.com/soobinseo/Transformer-TTS
