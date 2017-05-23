import numpy as np
from keras import backend as K
from keras import metrics
from keras.models import Model
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter

from vgg16_avg import VGG16_Avg


rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
deproc = lambda x,s: np.clip(x.reshape(s)[:, :, :, ::-1] + rn_mean, 0, 255)


class Evaluator(object):
    def __init__(self, f, shp):
        self.f = f 
        self.shp = shp
        
    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)


def solve_image(eval_obj, niter, x, shp):
    for i in range(niter):
        x, min_val, info = fmin_l_bfgs_b(
            eval_obj.loss,
            x.flatten(),
            fprime=eval_obj.grads,
            maxfun=20
        )
        x = np.clip(x, -127, 127)
        print('Current loss value:', min_val)
        imsave('results/res_at_iteration_{}.png'.format(i), deproc(x.copy(), shp)[0])
    return x


def gram_matrix(x):
    # We want each row to be a channel, and the columns to be flattened x,y locations
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    # The dot product of this with its transpose shows the correlation 
    # between each pair of channels
    return K.dot(features, K.transpose(features)) / x.get_shape().num_elements()


def style_loss(x, targ):
    return metrics.mse(gram_matrix(x), gram_matrix(targ))


def random_image(shape):
    """Create a random image with dimensions equal to shape"""

    return np.random.uniform(-2.5, 2.5, shape) / 100


# load and process the content image
content_img = Image.open('suicide-girl-2.jpg')
content_arr = preproc(np.expand_dims(np.array(content_img), 0))
content_shape = content_arr.shape

# load and process the style image
style_img = Image.open('snake.jpg')
style_img = style_img.resize((content_shape[2], content_shape[1]))
style_arr = preproc(np.expand_dims(style_img, 0)[:, :, :, :3])
style_shape = style_arr.shape


model = VGG16_Avg(include_top=False, input_shape=style_shape[1:])
outputs = {l.name: l.output for l in model.layers}
style_layers = [outputs['block{}_conv2'.format(o)] for o in range(1, 6)]
content_name = 'block4_conv2'
content_layer = outputs[content_name]

style_model = Model(model.input, style_layers)
style_targs = [K.variable(o) for o in style_model.predict(style_arr)]

content_model = Model(model.input, content_layer)
content_targ = K.variable(content_model.predict(content_arr))

# style_wgts = [0.05, 0.2, 0.2, 0.25, 0.3]
style_wgts = [0.02, 0.1, 0.1, 0.15, 0.15]

loss = sum(style_loss(l1[0], l2[0])*w
           for l1, l2, w in zip(style_layers, style_targs, style_wgts))
loss += metrics.mse(content_layer, content_targ) / 10
grads = K.gradients(loss, model.input)
transfer_fn = K.function([model.input], [loss]+grads)

evaluator = Evaluator(transfer_fn, style_shape)

iterations = 20
x = rand_img_100(style_shape)

x = solve_image(evaluator, iterations, x, style_shape)
