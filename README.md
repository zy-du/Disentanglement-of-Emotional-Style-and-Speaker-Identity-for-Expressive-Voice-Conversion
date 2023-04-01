## Disentanglement of Emotional Style and Speaker Identity for Expressive Voice Conversion

This is the implementation of our Interspeech 2022 paper "Disentanglement of Emotional Style and Speaker Identity for Expressive Voice Conversion". [Paper Link](https://arxiv.org/pdf/2110.10326.pdf)

## Requirements
* Python 3.8
* Pytorch 1.13.1 and Cuda 11.7 (**Note**: Other versions of PyTorch may also work, but have not been tested)
* [apex](https://github.com/NVIDIA/apex)
* Other requirements are listed in 'requirements.txt':

	pip install -r requirements.txt


## Training and inference:
*  Step1. Data preparation & preprocessing
1. Put [ESD](https://github.com/HLTSingapore/Emotional-Speech-Data) corpus under directory: 'Dataset/'
2. Feature (mel+lf0) extraction:

		python preprocess.py

*  Step2. model training:
	
		python train.py 


*  Step3. model testing:
  
		python lf0_normalization.py 
		python convert.py # config/convert.yaml need to be modified 
	
	
## Citation
If you use this code, please cite:
```
@inproceedings{du22c_interspeech,
  author={Zongyang Du and Berrak Sisman and Kun Zhou and Haizhou Li},
  title={{Disentanglement of Emotional Style and Speaker Identity for Expressive Voice Conversion}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={2603--2607},
  doi={10.21437/Interspeech.2022-10249}
}
```

## Acknowledgements:
The codes are based on VQMIVC:

https://github.com/Wendison/VQMIVC

Paper: [Link](https://arxiv.org/abs/2106.10132)


