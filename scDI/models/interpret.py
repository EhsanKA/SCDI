import numpy as np
import torch
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, GradientShap, InputXGradient, ShapleyValueSampling


def interpret_classes(model, x, prediction, method, num_classes=32):
    methods = {
        'IntGrad': IntegratedGradients,
        'DeepLift': DeepLift,
        'DeepLiftShape' : DeepLiftShap,
        'GradientShap' : GradientShap,
        'InputXGradient' : InputXGradient,
        'ShapleyValueSampling' : ShapleyValueSampling
    }
    im = methods[method](model)
    output = np.zeros_like(x)
    for i in range(num_classes):
        indices = np.where(prediction == i)[0]
        if indices.shape[0] > 1:
            attr = im.attribute(x[indices,], target=i)
            output[indices] = attr.detach().numpy()

    return np.abs(output)


def run_intMethods(adata, model, methods, num_classes=32):
    x_train = torch.from_numpy(data)
    prediction = model.predict(x_train)
    for method in methods:
        output = np.zeros_like(adata.X)
        for i in range(int(adata.X.shape[0] / 500)):
            print(method)
            start, end = i * 500, (i + 1) * 500
            output[start: end] = interpret_classes(model, x_train[start: end], prediction[start: end], method, num_classes)

        start = int(adata.X.shape[0] / 500) * 500
        output[start:, ] = interpret_classes(model, x_train[start:], prediction[start:], method, num_classes)
        adata.layers[method] = output
