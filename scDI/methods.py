from deepexplain.tensorflow import DeepExplain
from keras.models import Model
import keras.backend as K
import numpy as np
from collections import defaultdict
import innvestigate
import innvestigate.utils
import shap
import pandas as pd
import scanpy as sc


SUPPORTED_FRAMEWORKS = ["tensorflow", "keras", "pytorch"]
KERAS_SUPPORTED_METHODS = ["deeplift", "elrp", "intgrad", "smoothgrad", "deep_taylor"]
TENSORFLOW_SUPPORTED_METHODS = ["deeplift", "elrp", "intgrad"]
PYTORCH_SUPPORTED_METHODS = ["deep_explainer", "gradient_explainer"]


def interpret(model, adata, method, framework="keras", index=None, **kwargs):
    """

    :param model: Deep Neural Network model (Keras)
    :param adata: AnnData
    :param method: one of ["deeplift", "elrp", "intgrad", "smoothgrad", "deep_taylor"]
    :param framework: one of ["tensorflow", "keras", "pytorch"] but pytorch is not ready at the moment
    :param index: The final node of the model (predicted node)
    :param kwargs: kwargs
    :return:
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise Exception("Unsupported framework.")

    if not(method in adata.uns.keys()):
        adata.uns[method] = {}

    if framework == "keras":
        if method not in KERAS_SUPPORTED_METHODS:
            raise Exception("Unsupported method.")

        output_size = model.layers[-1].output.shape[-1]
        mask = np.zeros((1, output_size))

        def _DE():

            with DeepExplain(session=K.get_session()) as de:

                X = model.layers[0].input
                reconstructed_model = Model(inputs=X, outputs=model.layers[-1].output)
                T = reconstructed_model(X)

                if index is None:
                    for i in range(output_size):
                        mask[0][i] = 1
                        att = de.explain(method, T * mask, X, adata.X)
                        adata.uns[method][i] = att
                        mask[0][i] = 0

                else:
                    assert index >= 0
                    if index >= output_size:
                        raise Exception("Invalid index")

                    mask[0][index] = 1
                    att = de.explain(method, T * mask, X, adata.X)
                    adata.uns[method][index] = att

        def _INN():
            analyzer = innvestigate.create_analyzer(method, model, neuron_selection_mode="index")

            if index is None:
                for i in range(output_size):
                    att = analyzer.analyze(adata.X, i)
                    adata.uns[method][i] = att

            else:
                assert index >= 0
                if index >= output_size:
                    raise Exception("Invalid index")
                att = analyzer.analyze(adata.X, index)
                adata.uns[method][index] = att

        if method in ["deeplift", "elrp", "intgrad"]:
            return _DE()
        else:
            return _INN()



    elif framework == "tensorflow":
        if method not in TENSORFLOW_SUPPORTED_METHODS:
            raise Exception("Unsupported method.")

        sess = kwargs['session']
        X = kwargs['input_tensor']
        T = kwargs['target_tensor']

        with DeepExplain(session=sess) as de:

            output_size = T.shape[-1]
            mask = np.zeros((1, output_size))

            if index is None:
                for i in range(output_size):
                    mask[0][i] = 1
                    att = de.explain(method, T * mask, X, adata.X)
                    adata.uns[method][i] = att
                    mask[0][i] = 0
            else:
                assert index >= 0
                if index >= output_size:
                    raise Exception("Invalid index")

                mask[0][index] = 1
                att = de.explain(method, T * mask, X, adata.X)
                adata.uns[method][index] = att

    else:
        if method not in TENSORFLOW_SUPPORTED_METHODS:
            raise Exception("Unsupported method.")

        PYTORCH_METHODS = {'deep_explainer': shap.DeepExplainer, 'gradient_explainer': shap.GradientExplainer}
        fraction = kwargs['fraction']
        background = adata.X[np.random.choice(range(adata.X.shape[0]), int(adata.X.shape[0] * fraction), replace=False),
                     :]
        analyzer = PYTORCH_METHODS[method]

        attributions = analyzer(model, background).shap_values(adata.X)
        output_size = model.layers[-1].output.shape[-1]

        if index is None:
            for i, att in enumerate(attributions):
                adata.uns[method][i] = att
        else:
            assert index >= 0
            att = attributions[index]
            adata.uns[method][index] = att


def interpret_classes(model, adata, method, prediction, label_key='cell_label', framework="keras"):
    '''
    calling interpret for a supervised task like classification.
    :param model: Neural network model
    :param adata: annotation datain
    :param method: interpretability method name
    :param label_key: column name of labels in adata
    :param framework: Neural network framework like: Keras, Tensorflow and pytorch
    :return: numpy array with same shape to adata.X
    '''

    Y = pd.get_dummies(adata.obs[label_key])
    output = np.zeros_like(adata.X)

    for i in range(len(Y.columns)):
        indices = np.where(prediction == i)[0]
        if indices.shape[0] >1:
            # print(indices.shape)
            new_adata = adata[indices,].copy()
            interpret(model, new_adata, method, framework=framework, index=i)
            output[indices] = new_adata.uns[method][i]
    return output

def adata_top_genes(adata, method, index):
    """
    extracting most frequent genes for the selected method and index
    :param adata: AnnData
    :param method: one of interpretability methods
    :param index: index of output of deep model
    :return:
    """

    result = adata.uns[method][index]
    argResult = np.matrix.argsort(np.abs(result), axis=1)[:, -100:]
    uniq, cnts = np.unique(argResult, return_counts=True)
    geneDic = dict(zip(uniq, cnts))
    topGenesArgs = sorted(geneDic, key=geneDic.get, reverse=True)[:100]
    topGenes = adata.var.index[topGenesArgs]
    if not ('top_genes' in adata.layers.keys):
        adata.uns['top_genes'] = {}

    adata.uns['top_genes'][index] = topGenes


def remove_softmax(model):
    """
    Removes the softmax layer of a Keras model.
    :param model: deep model(Keras)
    :return:
    """


    return innvestigate.utils.model_wo_softmax(model)


def run_intMethods(adata, model, label_key, methods):
    """
    generates all attributions of all mentioned interpretability methods
    :param adata: AnnData
    :param model: Deep model
    :param label_key: name of columns which contains labels
    :param methods: list of interpretability methods
    :return:
    """
    y_pred = model.predict(adata.X)
    y_pred = y_pred.argmax(axis=1)
    for method in methods:
        print(method)

        if method in ['deep_taylor', 'smoothgrad']:
            adata.layers[method] = interpret_classes(remove_softmax(model), adata, method, y_pred, label_key)
            # interpret(remove_softmax(model), adata, method)

        else:
            adata.layers[method] = interpret_classes(model, adata, method, y_pred, label_key)
            # interpret(model, adata, method)


