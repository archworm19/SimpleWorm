"""Integration tests = training tests"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from Models.SoftTree.forest import build_forest
from Models.SoftTree.simple_predictors import LinearPred
from Models.SoftTree.layers import LayerFactoryBasic
from Models.SoftTree.objective_funcs import MultinomialLoss, BinaryLoss, QuantileLoss
from Models.SoftTree.assembled_models import GatedLossModel, NoGateModel
from Models.SoftTree.train_model import DataPlan, train, _gather_data, TrainStep
from Models.SoftTree.data_weight_generators import NullDWG
import pylab as plt
from tensorflow.keras.optimizers import SGD
from numpy.random import default_rng


def _build_big_model():
    # use this for testing assembly
    prediction_dim = 6
    layer_width = 2
    base_models = 2
    models_per_base = 4
    xshape = [5, 5]  # input from 5x5 matrix space
    rng = default_rng(42)

    layer_factory = LayerFactoryBasic(base_models, models_per_base, xshape, layer_width, rng)

    # forest = gating model
    depth = 2
    forest, forest_width, _ = build_forest(depth, layer_factory, 0., 0.)

    # simple prediction model
    base_models_pred = forest.get_num_state() * (base_models * models_per_base)
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dim, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, forest.get_num_state())

    # objective function:
    norm_dim = 3  # which dim of tensor contains the logits
    obj_func = MultinomialLoss(norm_dim)

    # whole model
    GLM = GatedLossModel(forest, pred_model, obj_func)
    # TODO: is this actually right?
    assert(len(GLM.get_trainable_weights()) == 15)
    # run stuff thru model
    batch_size = 64
    xshape_dat = [batch_size] + xshape
    x = tf.constant(npr.rand(*xshape_dat), dtype=tf.float32)
    model_predz = GLM.get_preds([x])
    assert(len(model_predz) == 2)
    assert(model_predz[0].shape == (batch_size, base_models * models_per_base, forest.get_num_state()))
    assert(model_predz[1].shape == (batch_size, base_models * models_per_base,
                                    forest.get_num_state(), prediction_dim))

    # y dictated by objective function
    y = npr.randint(0, 6, (batch_size,))
    y = tf.constant(y, dtype=tf.int32)
    loss_sample = GLM.loss_samples_noreg([x], y)
    assert(loss_sample.shape == (batch_size, base_models * models_per_base))
    return GLM


def _build_binary_classifier_model(forest_penalty: float = 0.):
    prediction_dims = 1  # binary
    layer_width = 2
    base_models = 2
    models_per_base = 4
    xshape = [2]
    rng = default_rng(42)

    layer_factory = LayerFactoryBasic(base_models, models_per_base, xshape, layer_width, rng)

    # forest = gating model
    depth = 1
    forest, forest_width, _ = build_forest(depth, layer_factory, forest_penalty, 0.)

    # simple prediction model
    base_models_pred = forest.get_num_state() * (base_models * models_per_base)
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dims, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, forest.get_num_state())

    # objective function = binary class
    # NOTE: pred_model --> batch_size x num_model x num_state x prediction dim (total_dims = 4)
    obj_func = BinaryLoss(4, 3)  # 3rd dim = (len=1) classifier dimension
    return GatedLossModel(forest, pred_model, obj_func)


def _build_null_binary_classifier():
    # NOTE: forest doesn't matter here...
    prediction_dims = 1  # binary
    base_models = 2
    models_per_base = 4
    xshape = [2]
    rng = default_rng(42)

    # simple prediction model
    base_models_pred = base_models * models_per_base
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dims, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, 1)

    # objective function = binary class
    # NOTE: pred_model --> batch_size x num_model x num_state x prediction dim (total_dims = 4)
    obj_func = BinaryLoss(4, 3)  # 3rd dim = (len=1) classifier dimension
    return NoGateModel(pred_model, obj_func)


def _build_null_quantile_loss():
    # NOTE: forest doesn't matter here...
    prediction_dims = 5  # 5 taus
    base_models = 2
    models_per_base = 4
    xshape = [2]
    rng = default_rng(42)

    # simple prediction model
    base_models_pred = base_models * models_per_base
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dims, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, 1)

    # objective function = quantile regression:
    # NOTE: pred_model --> batch_size x num_model x num_state (1 here) x prediction dim (total_dims = 4)
    taus = tf.constant([.1, .25, .5, .75, .9])
    obj_func = QuantileLoss(4, 3, taus)
    return NoGateModel(pred_model, obj_func)


def test_binary_class():
    GLM = _build_binary_classifier_model()

    # easy classification problem:
    # > if x1 + x2 > 1.0 --> yea
    batch_size = 128
    x = npr.rand(batch_size, 2)
    y = 1 * (np.sum(x, axis=1) > 1.)
    # convert to dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32),
                                                        y.astype(np.float32),
                                                        np.arange(batch_size))).batch(32)
    data_plan = DataPlan([0], 1, 2)
    dw_gen = NullDWG(8)
    # train!
    train_step = TrainStep(GLM, SGD())
    tr_losses = train(train_dataset, data_plan, dw_gen, train_step,
                        15, 15 * [1.])
    # plt results
    plt.figure()
    plt.plot(tr_losses)
    plt.figure()
    for batch in train_dataset:
        x, y, _dw = _gather_data(batch, data_plan, dw_gen)
        # average estimates across models and state
        # NOTE: these are logits
        logit_preds = np.mean(GLM.get_preds(x)[1].numpy(), axis=(1, 2, 3))
        plt.scatter(logit_preds, y.numpy(), c='b')
    plt.xlabel("model prediction logit")
    plt.ylabel("truth")
    plt.show()


def test_xor():
    # XOR binary classification problem
    # --> should require depth
    GLM = _build_binary_classifier_model()
    NullM = _build_null_binary_classifier()

    # XOR classification problem
    # > if x1 > 0.5 XOR x2 > 0.5
    batch_size = 128
    x = npr.rand(batch_size, 2)
    y = 1 * (np.logical_xor(x[:,0] > 0.5, x[:,1] > 0.5))
    # convert to dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32),
                                                        y.astype(np.float32),
                                                        np.arange(batch_size))).batch(32)
    data_plan = DataPlan([0], 1, 2)
    dw_gen = NullDWG(8)  # there are 8 models (2 parallel sets)
    # train null model
    null_train_step = TrainStep(NullM, SGD())
    tr_losses_null = train(train_dataset, data_plan, dw_gen, null_train_step,
                           300, 300 * [1.])
    # train full model
    full_train_step = TrainStep(GLM, SGD(0.05))
    tr_losses_full = train(train_dataset, data_plan, dw_gen, full_train_step,
                           300, [0.95 ** z for z in range(300)])

    plt.figure()
    plt.plot(tr_losses_null, c='b')
    plt.plot(tr_losses_full, c='r')
    plt.legend(["Null", "Full"])

    plt.figure()
    for batch in train_dataset:
        x, y, _dw = _gather_data(batch, data_plan, dw_gen)
        # average estimates across models and state
        # NOTE: these are logits

        # scaled predictions for forest
        [gates, preds] = GLM.get_preds(x)
        # --> batch_size x models
        w_preds = np.sum(gates * np.sum(preds, axis=3), axis=2)
        # average across models --> batch_size
        logit_preds_full = np.mean(w_preds, axis=1)
        plt.scatter(logit_preds_full, y.numpy(), c='r')

        # nulls
        logit_preds_null = np.mean(NullM.get_preds(x)[0].numpy(), axis=(1, 2, 3))
        plt.scatter(logit_preds_null, y.numpy(), c='b')

    plt.xlabel("model prediction logit")
    plt.ylabel("truth")
    plt.show()


def test_quantile_regression():
    # quantile regression without gating model
    # taus = .1, .25, .5, .75, .9
    taus = [.1, .25, .5, .75, .9]
    M = _build_null_quantile_loss()

    # don't care about x, just have it learn offsets
    N = 128
    x = np.zeros((N, 2))
    y = np.arange(128)
    y_true = y
    # convert to dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32),
                                                        y.astype(np.float32),
                                                        np.arange(N))).batch(32)
    data_plan = DataPlan([0], 1, 2)
    dw_gen = NullDWG(8)  # there are 8 models (2 parallel sets)
    # train null model
    null_train_step = TrainStep(M, SGD(0.1))
    tr_losses_null = train(train_dataset, data_plan, dw_gen, null_train_step,
                           30, 30 * [1.])

    plt.figure()
    plt.plot(tr_losses_null)

    # NOTE: since no relevant x --> will get constant prediction!
    v = []
    for batch in train_dataset:
        x, _y, _dw = _gather_data(batch, data_plan, dw_gen)
        # average estimates across models and state
        # NOTE: these are logits
        # nulls
        logit_preds_null = np.mean(M.get_preds(x)[0].numpy(), axis=(1, 2))
        v.append(logit_preds_null)
    v = np.mean(np.vstack(v), axis=0)
    y_true = [np.percentile(y_true, tau * 100) for tau in taus]
    plt.figure()
    plt.scatter(v, y_true)
    plt.show()



# TODO: test other types of models...


if __name__ == "__main__":
    _build_big_model()
    test_binary_class()
    test_xor()
    test_quantile_regression()