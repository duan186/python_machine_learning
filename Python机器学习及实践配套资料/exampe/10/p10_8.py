from sklearn.metrics import zero_one_loss

y_true=[1,1,1,1,1,0,0,0,0,0]
y_pred=[0,0,0,1,1,1,1,1,0,0]
print('zero_one_loss<fraction>:',zero_one_loss(y_true,y_pred,normalize=True))
print('zero_one_loss<num>:',zero_one_loss(y_true,y_pred,normalize=False))
