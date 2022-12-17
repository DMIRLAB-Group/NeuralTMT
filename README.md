# NeuralTMT
The code and dataset for our NN 2022 paper: Factorizing Time-heterogeneous Markov Transition for Temporal Recommendation(https://www.sciencedirect.com/science/article/abs/pii/S0893608022004737). We have implemented our methods in Pytorch.

## Dependencies

- Python 3.8
- torch 1.10.1

## Usage 

### Generate data

You need to run the file ```pre_data.py``` to generate the data format needed for our model.
For example:
```
u1:[u1,[i_1,i_2],[i_3,i_4,i_5],[i_3,i_8],[i_0,i_5],[i_1,i_2],[i_4,i_5],[i_6,i_7],[i_1,i_2]]
u2:[u2,[i_1,i_4],[i_5,i_2],[i_7,i_9],[i_10,i_21],[i_1,i_2],[i_3,i_4,i_5],[i_6,i_7],[i_1,i_2]]
...
```


### Training and Testing 

Then you can run the file ```main.py``` to train and test our model. 


## Cite
If you want to use our codes in your research, please cite:
```
@ARTICLE
  article{wen2022factorizing,
  title={Factorizing time-heterogeneous Markov transition for temporal recommendation},
  author={Wen, Wen and Wang, Wencui and Hao, Zhifeng and Cai, Ruichu},
  journal={Neural Networks},
  year={2022},
  publisher={Elsevier}
}
