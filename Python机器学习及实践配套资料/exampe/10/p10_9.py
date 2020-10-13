from sklearn.metrics import log_loss
y_true=[1,1,1,0,0,0]
y_pred=[
    [0.1,0.9],
    [0.2,0.8],
    [0.3,0.7],
    [0.7,0.3],
    [0.8,0.2],
    [0.9,0.1]
]
print('log_loss<average>:',log_loss(y_true,y_pred,normalize=True))
print('log_loss,total.:',log_loss(y_true,y_pred,normalize=False))
