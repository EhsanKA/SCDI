import tensorflow as tf
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from cleverhans.attacks import FastGradientMethod, ElasticNetMethod, BasicIterativeMethod, MomentumIterativeMethod, CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper


def adversarial_attacks(model, attack_method, x, target, y=None, attack_params=None, num_classes=8, targeted_attack=True):
    """

    :param model: Deep Neural Network model (Keras)
    :param attack_method: an string from [FastGradientMethod, ElasticNetMethod, BasicIterativeMethod, MomentumIterativeMethod, CarliniWagnerL2
    :param x: e.g. x_tring
    :param target: target label
    :param y: predicted labels
    :param attack_params: relevance parameters to each attack method
    :param num_classes: number of classes
    :param targeted_attack: choose the attack generate samples with specific target or not
    :return:
    """
    model_wrapper = KerasModelWrapper(model)

    sess = K.get_session()
    y_target = tf.placeholder(tf.int32, shape=[None, num_classes])

    if attack_method == 'FastGradientMethod':
        method = FastGradientMethod(model_wrapper, sess=sess)
        if attack_params is None:
            attack_params = {'clip_min': 0.,
                             'clip_max': float(np.max(x)),
                             }
        if targeted_attack:
            attack_params['y_target'] = y_target

        else:
            attack_params['y'] = y_target

        adv_x, adv_y = adv_example(model, method, attack_params)

        if targeted_attack:
            target_labels = to_categorical(np.zeros((x.shape[0],)) + target, num_classes=num_classes)
        else:
            target_labels = y

        x_adversarial, y_adversarial = sess.run([adv_x, adv_y],
                                                feed_dict={model.layers[0].input: x,
                                                           y_target: target_labels})
        return x_adversarial, y_adversarial

    if attack_method == 'BasicIterativeMethod':
        method = BasicIterativeMethod(model_wrapper, sess=sess)
        if attack_params is None:
            attack_params = {'clip_min': 0.,
                             'clip_max': float(np.max(x)),
                             'nb_iter': 50,
                             'eps_iter': .01,
                             }
        if targeted_attack:
            attack_params['y_target'] = y_target

        else:
            attack_params['y'] = y_target

        adv_x, adv_y = adv_example(model, method, attack_params)

        if targeted_attack:
            target_labels = to_categorical(np.zeros((x.shape[0],)) + target, num_classes=num_classes)
        else:
            target_labels = y

        x_adversarial, y_adversarial = sess.run([adv_x, adv_y],
                                                feed_dict={model.layers[0].input: x,
                                                           y_target: target_labels})
        return x_adversarial, y_adversarial

    if attack_method == 'MomentumIterativeMethod':
        method = MomentumIterativeMethod(model_wrapper, sess=sess)
        if attack_params is None:
            attack_params = {'clip_min': 0.,
                             'clip_max': float(np.max(x)),
                             'nb_iter': 60,
                             'eps_iter': .01,
                             }
        if targeted_attack:
            attack_params['y_target'] = y_target

        else:
            attack_params['y'] = y_target

        adv_x, adv_y = adv_example(model, method, attack_params)

        if targeted_attack:
            target_labels = to_categorical(np.zeros((x.shape[0],)) + target, num_classes=num_classes)
        else:
            target_labels = y

        x_adversarial, y_adversarial = sess.run([adv_x, adv_y],
                                                feed_dict={model.layers[0].input: x,
                                                           y_target: target_labels})
        return x_adversarial, y_adversarial

    if attack_method == 'CarliniWagnerL2':
        method = CarliniWagnerL2(model_wrapper, sess=sess)
        if attack_params is None:
            attack_params = {'clip_min': 0.,
                             'clip_max': float(np.max(x)),

                             }
        if targeted_attack:
            attack_params['y_target'] = y_target

        else:
            attack_params['y'] = y_target

        adv_x, adv_y = adv_example(model, method, attack_params)

        if targeted_attack:
            target_labels = to_categorical(np.zeros((x.shape[0],)) + target, num_classes=num_classes)
        else:
            target_labels = y

        x_adversarial, y_adversarial = sess.run([adv_x, adv_y],
                                                feed_dict={model.layers[0].input: x,
                                                           y_target: target_labels})
        return x_adversarial, y_adversarial

    if attack_method == 'ElasticNetMethod':
        method = ElasticNetMethod(model_wrapper, sess=sess)
        if attack_params is None:
            attack_params = {'clip_min': 0.,
                             'clip_max': float(np.max(x)),
                             }
        if targeted_attack:
            attack_params['y_target'] = y_target

        else:
            attack_params['y'] = y_target

        adv_x, adv_y = adv_example(model, method, attack_params)

        if targeted_attack:
            target_labels = to_categorical(np.zeros((x.shape[0],)) + target, num_classes=num_classes)
        else:
            target_labels = y

        x_adversarial, y_adversarial = sess.run([adv_x, adv_y],
                                                feed_dict={model.layers[0].input: x,
                                                           y_target: target_labels})
        return x_adversarial, y_adversarial


def adv_example(model, attack_method, attack_params):
    """

    :param model: Deep Neural Network model (Keras)
    :param attack_method: an string from [FastGradientMethod, ElasticNetMethod, BasicIterativeMethod, MomentumIterativeMethod, CarliniWagnerL2]
    :param attack_params: relevance parameter to each attack as dictionary
    :return:
    """
    x_adv = attack_method.generate(model.layers[0].input, **attack_params)
    x_adv = tf.stop_gradient(x_adv)

    preds_adv = model(x_adv)

    return x_adv, preds_adv


# todo generating array like x all with the required targets!