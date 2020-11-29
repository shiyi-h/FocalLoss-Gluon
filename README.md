# FocalLoss-Gluon
Recently, I encountered the problem of unbalanced sample categories when doing natural language text classification tasks.</br>
Refer to Kaiming He's FocalLoss and use mxnet's gluon API to write a gluon version of FocalLoss.

# Main Params
- gamma: default 2
- alpha: float or list or np.ndarray or nd.NDArray, default 0.25
  - If you want to set a weight for each label category, you only need to pass in a list (one-dimensional matrix) of the same length as the number of categories, and each value in the list (one-dimensional matrix) represents the weight of the label.
  - Else, each label's weight will be set 0.25

# How to use
x = mx.nd.array([[1,2,3],[4,5,6],[9,8,7]])  

label = mx.nd.array([[1],[2],[0]])  

alpha = [1,2,3]  

loss = FocalLoss(alpha=alpha)  

loss.hybridize()  

print(loss(x,label))
