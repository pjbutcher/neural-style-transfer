import argparse
import sys
import numpy as np
from keras import backend as K
from keras import metrics
from keras.models import Model
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
from vgg16_avg import VGG16_Avg


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
        imsave(
            'results/res_at_iteration_{}.png'.format(i), 
            deprocess(x.copy(), rn_mean, shp)[0]
        )
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


def preprocess(img_arr, img_mean):
    return (img_arr - img_mean)[:, :, :, ::-1]


def deprocess(img_arr, img_mean, shape):
    return np.clip(img_arr.reshape(shape)[:, :, :, ::-1] + img_mean, 0, 255)


# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('content')
parser.add_argument('style')
parser.add_argument('--output_final')
parser.add_argument('--output_iters')
parser.add_argument('--weights', nargs='*', default=[0.05, 0.2, 0.2, 0.25, 0.3])
parser.add_argument('--iters', default=20)
parser.add_argument('--content_layer', default='block4_conv2')
parser.add_argument('--style_layers', default=[1, 5, 'conv2'])
args = parser.parse_args()


if len(args.weights) != 5:
    print('weights argument requires 5 values.')
    sys.exit()

style_wgts = list(map(float, args.weights))
iterations = int(args.iters)
content_name = args.content_layer

# mean from VGG authors used to preprocess images
rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)

# load and process the content image
content_img = Image.open(args.content)
content_arr = preprocess(np.expand_dims(np.array(content_img), 0), rn_mean)
content_shape = content_arr.shape

# load and process the style image
style_img = Image.open(args.style)
style_img = style_img.resize((content_shape[2], content_shape[1]))
style_arr = preprocess(np.expand_dims(style_img, 0)[:, :, :, :3], rn_mean)
style_shape = style_arr.shape

# instantiate vgg model
model = VGG16_Avg(include_top=False, input_shape=style_shape[1:])

# get a dictionary of the models name:output
outputs = {l.name: l.output for l in model.layers}

# use block{1-5}_conv2 to recreate style
sblock = args.style_layers[0]
eblock = args.style_layers[1]
block_layer = args.style_layers[2]
style_layers = [outputs['block{}_{}'.format(o, block_layer)]
                for o in range(sblock, eblock+1)]
# style_layers = [outputs['block{}_conv2'.format(o)] for o in range(1, 6)]

# layer to recreate content
content_layer = outputs[content_name]

style_model = Model(model.input, style_layers)
style_targs = [K.variable(o) for o in style_model.predict(style_arr)]

content_model = Model(model.input, content_layer)
content_targ = K.variable(content_model.predict(content_arr))


loss = sum(style_loss(l1[0], l2[0])*w
           for l1, l2, w in zip(style_layers, style_targs, style_wgts))
loss += metrics.mse(content_layer, content_targ) / 10
grads = K.gradients(loss, model.input)
transfer_fn = K.function([model.input], [loss]+grads)

evaluator = Evaluator(transfer_fn, style_shape)

styled_img = random_image(style_shape)
for i in range(iterations):
    styled_img, min_val, info = fmin_l_bfgs_b(
        evaluator.loss,
        styled_img.flatten(),
        fprime=evaluator.grads,
        maxfun=20
    )
    styled_img = np.clip(styled_img, -127, 127)
    print('Current loss value: {} (iteration {}/{})'.format(
        min_val, i+1, iterations))

    if args.output_iters is not None:    
        imsave(
            '{}/iteration_{}.png'.format(args.output_iters, i),
            deprocess(styled_img.copy(), rn_mean, style_shape)[0]
        )

if args.output_final is not None:
    imsave(
        '{}.png'.format(args.output_final),
        deprocess(styled_img.copy(), rn_mean, style_shape)[0]
    )

