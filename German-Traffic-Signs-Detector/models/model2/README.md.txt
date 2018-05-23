Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to
 handle multiple classes. In logistic regression we assumed that the labels were binary: y(i)?{0,1}.
Softmax regression allows us to handle y(i)?{1,…,K} where K is the number of classes.
With this we can estimate the probability that P(y=k|x) for each value of k=1,…,K
A diference with the model 1 its that in this case there is a target vector (instead of a target value!) 
composed of only zeros and ones where only correct label is set as 1, this vector is called one hot encoded target vector

