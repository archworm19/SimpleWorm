"""Integration tests = training tests"""
import numpy as np
import numpy.random as npr
import tensorflow as tf
from Models.SoftTree.forest import build_forest
from Models.SoftTree.simple_predictors import LinearPred
from Models.SoftTree.layers import LayerFactoryBasic
from Models.SoftTree.objective_funcs import MultinomialLoss, BinaryLoss
from Models.SoftTree.assembled_models import GatedLossModel
from Models.SoftTree.train_model import DataPlan, train, _gather_data
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


def _build_binary_classifier_model():
    # NOTE: forest doesn't matter here...
    prediction_dims = 1  # binary
    layer_width = 2
    base_models = 2
    models_per_base = 4
    xshape = [2]
    rng = default_rng(42)

    layer_factory = LayerFactoryBasic(base_models, models_per_base, xshape, layer_width, rng)

    # forest = gating model
    depth = 1
    forest, forest_width, _ = build_forest(depth, layer_factory, 0., 0.)

    # simple prediction model
    base_models_pred = forest.get_num_state() * (base_models * models_per_base)
    layer_factory_pred = LayerFactoryBasic(base_models_pred, 1, xshape, prediction_dims, rng)
    pred_model = LinearPred(layer_factory_pred.build_layer(), base_models*models_per_base, forest.get_num_state())

    # objective function = binary class
    # NOTE: pred_model --> batch_size x num_model x num_state x prediction dim (total_dims = 4)
    obj_func = BinaryLoss(4, 3)  # 3rd dim = (len=1) classifier dimension
    return GatedLossModel(forest, pred_model, obj_func)


def test_binary_class():
    GLM = _build_binary_classifier_model()

    # easy classification problem:
    # > if x1 + x2 > 1.0 --> yea
    batch_size = 128
    x = npr.rand(batch_size, 2)
    y = 1 * (np.sum(x, axis=1) > 1.)
    data_weights = np.ones((batch_size, 8))  # there are 8 models (2 parallel sets)
    # convert to dataset:
    train_dataset = tf.data.Dataset.from_tensor_slices((x.astype(np.float32),
                                                        y.astype(np.float32),
                                                        data_weights.astype(np.float32))).batch(32)
    data_plan = DataPlan([0], 1, 2)
    # train!
    tr_losses = train(train_dataset, data_plan, GLM, SGD(),
                        15, 15 * [1.])
    # TODO: move this code elsewhere:
    plt.figure()
    plt.plot(tr_losses)
    plt.figure()
    for batch in train_dataset:
        x, y, _dw = _gather_data(batch, data_plan)
        # average estimates across models and state
        # NOTE: these are logits
        logit_preds = np.mean(GLM.get_preds(x)[1].numpy(), axis=(1, 2, 3))
        plt.scatter(logit_preds, y.numpy(), c='b')
    plt.xlabel("model prediction logit")
    plt.ylabel("truth")
    plt.show()



# TODO: test other types of models...


if __name__ == "__main__":
    _build_big_model()
    test_binary_class()