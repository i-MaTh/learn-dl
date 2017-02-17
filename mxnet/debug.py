import mxnet as mx

input_x = mx.sym.Variable('input_x')
embed_layer = mx.sym.Embedding(data=input_x, input_dim=26, output_dim=10, name='vocab_embed')

# infer_shape
input_shape = {'input_x' : (2,1)} # choose it by yourself.
print embed_layer.infer_shape(**input_shape) 
