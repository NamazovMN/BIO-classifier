# BIO-classifier

## Table Of Contents
* Project Description
* Data Processing
* Idea of solution and Approach
* Results and Comparison

## Project Description
Main idea of the project is to classify tokens in argumentative texts into following labels: B-Beginning, I-Intermediate, O-Out. What do they mean? These labels basically stand for the position of the tokens in the *Argumentative* part of the given texts. 

## Data Processing
Since classification only based on BIO labels does not provide *sufficient* results, another dimension for the task was introduced. I used POS tags as another dimension for the given task. In order to achieve good results the following steps were followed:
* Raw data were retrieved and vocabulary of the main task was generated
* All sentences were encoded according to the vocabulary, which was made based on train dataset
* Given sentences were tokenized and pos tagged thanks to NLTK pos-tag method
* As a result we have 2 label encoding. In other words, we will have 2 parallel branches in the model of classification task. It can be seen as multi-task classification
* What we got: Vocabulary of tokens, BIO label encoding, POS-tags label encoding

Following results can be shown as an example:
![Figure 1: Samples from Vocabulary and Label Encodings](https://github.com/NamazovMN/BIO-classifier/blob/master/Screenshot%20from%202022-10-02%2014-07-49.png)

## Idea of Solution and Approach
Bi-LSTM model was used as a classification model, since it analyzes sequences in both directions (left to right, right to left). However, it was obvious after running model only with BIO labels, model hardly beat the baseline result (F1(0.54)). Thus, I decided to use POS tags as another branch of the model. That is why it is called parallel model. Model was pushed to concentrate not only BIO labels, but also POS tags of the tokens that this extra information lead us to obtain sufficient results.

More details: Two channels are used to feed the model. It can be seen from the following figure easily
Note: Figure is prepared, yet!

## Results and Comparison:
As a result we trained model with 2 different learning rates, which  are 1e-4 and 1e-5. While F1 score (macro) 0.75 was obtained with first approach, F1(0.73) was obtained by second approach. 
Loss and Accuracy graphs will be evaluated and added soon.

P.S. Dataset is not published because of privacy reasons However relevant academic paper link will be published soon, too.

*Thanks for your attention!*

*Author: Mahammad Namazov*
