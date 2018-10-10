# Conv_LSTM_TeleCommunications

This project is a practice of statistical analysis, python programming, and application in deep learning techniques. And all the work is done by myself. Please feel free to contact me if you have any questions or advice. Also, please give me a star if you think the project is helpful, thanks!

## Goal
1. Analyze the statistical attributes of telecommunication data, like autocorrelations, Rolling Mean & Std Deviation, spatial information, temporial information, and analysis it deeply. 

2. Forecast future telecommunication trends using DEEP LEARNING techniques, then compared the method with LSTM, ConvLSTM, SVR and ARIMA; Coded with Python in TENSORFLOW framework and other Python packages

## Data Instruction and Statistical Analysis

This dataset provides information about the telecommunication activity over the city of Milano. You can get access to the data from [Telecommunications - SMS, Call, Internet - Milano](https://dandelion.eu/datamine/open-big-data/). The dataset divides a whole region into 100 * 100 grids, and each grid represents a small region in the whole map. In every grid, it contains 5 features: SMS in, SMS out, Call in, Call out, International Traffic. The telecommunication data is recorded every 10 minutes, lasting for 2 months. For statistical Analysis:

#### Basic Plot

Simply plot all of the data in all of the situation. For instance, 

1. plot 2 months' *Call In* data in every grid:

*Call in* data in grid 0-750:

![CallIn_1-750](https://i.imgur.com/mHtt8II.png)

*Call in* data in grid 751-1500:

![CallIn_751-1500](https://i.imgur.com/2H8mvPG.png)

2. In two weeks:

![CallIn_0-750_twoweeks](https://i.imgur.com/AY7CuZG.png)

3. In grid 1, plot combining 5 features:

![1_5features](https://i.imgur.com/xSe4nS5.png)

4. Call Out* data, record every 75 grids, and gather all of the pictures in one pdf file:

![CallOut_Every75Grids](https://i.imgur.com/LsO7uxr.jpg)

#### AutoCorrelation

For instance, implement autocorrelation on the *Call in* data in grid 0, considering **temporal** information:

![Autocorrelation_Cin\[0\]_temporal](https://i.imgur.com/87r8d6z.png)

Time = 0, implement autocorrelation on the *Call in* data, considering **spacial** information:

![Autocorrelation_Cin\[/,0\]_spacial](https://i.imgur.com/wXmQ1ej.png)

#### Rolling Mean & Stad Deviation

Time = 0, all of 100*100 grids' Rolling Mean & Stad Deviation for *Call In* data:

![RollingMean&StadDeviation_Cin\[/,0\]_spatial](https://i.imgur.com/tb2UFCv.png)

In grid 0, Rolling Mean & Stad Deviation for *Call In* data:

![RollingMean&StadDeviation_Cin\[0\]_temporal](https://i.imgur.com/7AaMSOX.png)


## Models
I use four models: **SVR**, **ARIMA**, **LSTM**, and **Convolutional LSTM** models to predict the future trendency of telecommunication data. 

### Conv_LSTM

The conv_LSTM architecture is proposed in this article: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/pdf/1506.04214.pdf). In this github project, there is also an implementation of conv_LSTM in tensorflow. 

1. Convolutional-LSTM-in-Tensorflow
An implementation of convolutional lstms in tensorflow. The code is written in the same style as the `basiclstmcell` function in tensorflow and was meant to test whether this kind of implementation worked. To test this method I applied it to the bouncing ball data set created by Ilya Sutskever in this paper [Recurrent Temporal Restricted Boltzmann Machine](http://www.uoguelph.ca/~gwtaylor/publications/nips2008/rtrbm.pdf). To add velocity information I made the x and y velocities correspond to the color of the ball. This was added so I could compare the results with just next frame prediction with straight convolutions.

2. Basics of how it works
All I really did was take the old lstm implementation and replace the fully connected layers with convolutional. I use the concatenated state implementation and concat on the depth dimension. I would like to redo the `rnn_cell.py` file in tensorflow with this method. This method first appears in the paper [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/pdf/1506.04214v2.pdf).

3. How well does it work!

Like LSTM, the predicted results are not as well as ARIMA.
