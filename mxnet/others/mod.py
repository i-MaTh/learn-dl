# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 15:51:57 2017

@author: i-MaTh
Descriptions: learning mxnet module
"""


import mxnet as mx


"""Construct network"""
def get_net(n_output):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    
    net = mx.sym.FullyConnected(data=data, num_hidden=256, name='fc1')
    #net = mx.sym.BatchNorm(data=net, fix_gamma=True)
    #net = mx.sym.LeakyRelu(data=net)
    #net = mx.sym.Dropout(data=net, p=0.5)
    net = mx.sym.Activation(data=net, act_type='relu', name='relu1')
    net = mx.sym.FullyConnected(data=net, num_hidden=128, name='fc2')
    net = mx.sym.Activation(data=net, act_type='tanh', name='relu2')
    net = mx.sym.FullyConnected(data=data, num_hidden=n_output, name='fc4')
    net = mx.sym.LinearRegressionOutput(data=net, label=label, name='lr')
    return net


if __name__=='__main__':
    batch_size = 64
    num_output = 10
    n_epoch = 10
    lr = 0.01    
    
    train_x = []
    train_y = []
    trainIter = mx.io.NDArrayIter(data = train_x, label = train_y, shuffle=True, batch_size = batch_size)
    network = get_net(num_output)
    mod = mx.mod.Module(symbol=network, context=mx.cpu())
    
    opt = 'sgd'
    opt_params={'learning_rate':lr, 'wd':0.0001}  
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
        
    mod.fit(train_data = trainIter,
            optimizer =opt,
            optimizer_params = opt_params,
            num_epoch = n_epoch,
            eval_metric='rmse',
            initializer = mx.init.Xavier(factor_type='in',magnitude=2.34),
            batch_end_callback=mx.callback.Speedometer(batch_size,10),
        )
        
    trainIter.reset()
    mod.predict(trainIter).asnumpy()
    






