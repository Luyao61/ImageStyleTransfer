
from keras.applications import vgg19
from keras import backend as K
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave


image_height = 400
image_width = 400
content_weight = 1


def preprocess_image(path):
    img = load_img(path, target_size=(image_height, image_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, image_height, image_width))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((image_height, image_width, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

content_image_path = "examples/golden_gate.jpg"
style_image_path = "examples/the-starry-night-1889.jpg"
content_image = K.variable(preprocess_image(content_image_path))
style_image = K.variable(preprocess_image(style_image_path))
combination_image = K.placeholder((1, image_height, image_width, 3))
input_tensor = K.concatenate([content_image,
                              style_image,
                              combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


def content_loss(content, combination):
    return K.sum(K.square(combination - content))

loss = K.variable(0.)
layer_features = outputs_dict['block4_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(content_image_features,
                                      combination_features)

dLoss_dx = K.gradients(loss, combination_image)
outputs = [loss]
if type(dLoss_dx) in {list, tuple}:
    outputs += dLoss_dx
else:
    outputs.append(dLoss_dx)
f_outputs = K.function([combination_image], outputs)









def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, image_height, image_width))
    else:
        x = x.reshape((1, image_height, image_width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
if K.image_dim_ordering() == 'th':
    x = np.random.uniform(0, 255, (1, 3, image_height, image_width)) - 128.
else:
    x = np.random.uniform(0, 255, (1, image_height, image_width, 3)) - 128.

for i in range(10):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = "result" + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
