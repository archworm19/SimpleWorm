"""Testing forest components"""
from typing import List
from Models.SoftTree import layers
from Models.SoftTree import forest
from Models.SoftTree import decoders
# from Models.SoftTree.assembled_models_em import GMMforestEM  # TODO: port to new interface
import Models.SoftTree.train_model as train_model
from Models.SoftTree.objective_funcs import BinaryLoss, MultinomialLoss, QuantileLoss
import tensorflow as tf
import numpy as np
import numpy.random as npr
import numpy.linalg as npla
from scipy.stats import multivariate_normal as m_normal


def test_layers():
    rng = npr.default_rng(42)

    num_base_model = 3
    models_per_base = 4
    batch_size = 2
    xshape = [4, 5]
    width = 3
    x = tf.ones([batch_size] + xshape)
    LFB = layers.LayerFactoryBasic(num_base_model, models_per_base, xshape, width, rng)
    layer1 = LFB.build_layer()
    layer2 = LFB.build_layer()

    # ensure layers are not identical == factory (not clone)
    assert(np.sum(layer1.eval(x).numpy() != layer2.eval(x).numpy()) >= 1)
    assert(np.shape(layer1.eval(x).numpy()) == (2, 12, 3))
    assert(np.shape(layer1._get_wop().numpy()) == tuple([1, num_base_model * models_per_base, width] + xshape))
    assert(np.shape(layer1.offset.numpy()) == (12, 3))
    # first 3 = batch_size, num_model, width (reduce across xshape)
    assert(layer1.reduce_dims == [3, 4])
    
    # low rank layers
    low_dim = 2
    LFLR = layers.LayerFactoryFB(num_base_model, models_per_base, xshape, width, low_dim, rng)
    layer3 = LFLR.build_layer()
    layer4 = LFLR.build_layer()
    assert(np.sum(layer3.eval(x).numpy() != layer4.eval(x).numpy()) >= 1)
    assert(np.shape(layer3.eval(x).numpy()) == (2, 12, 3))
    assert(np.shape(layer3.fb.numpy()) == tuple([num_base_model, 1, 1, low_dim] + xshape))
    assert(np.shape(layer3.w_shared.numpy()) == tuple([num_base_model, 1, width, low_dim]
                                                        + [1 for _ in xshape]))
    assert(np.shape(layer3.w_spread.numpy()) == tuple([num_base_model, models_per_base, width, low_dim]
                                                        + [1 for _ in xshape]))

    # test that parallel model flattening is done correctly:
    # raw_w = base_models x models_per_base x width x xdims
    # while wop = 1 x base_models * models_per_base x width x xdims
    raw_w = tf.reduce_sum(layer4.fb * layer4.w_shared + layer4.fb * layer4.w_spread, axis=3).numpy()
    wop = layer4._get_wop().numpy()
    for i in range(num_base_model):
        for j in range(models_per_base):
            raw_np = raw_w[i,j]
            w_np = wop[0,i*models_per_base + j]
            assert(np.sum(raw_np - w_np) < .00001)

    # testing multilayer:
    multi_layer = layers.LayerMulti([layer1, layer3])
    assert(np.shape(multi_layer.eval([x,x]).numpy()) == np.shape(layer1.eval(x).numpy()))


def _get_layer_ws(node: forest.ForestNode, w_list: List):
    w_list.append(node.layer._get_wop()[0])
    for child in node.children:
        _get_layer_ws(child, w_list)


def _build_forest():
    depth = 3  # 3 levels (2 levels with children)
    num_base_models = 3
    models_per_base = 5
    xshape = [12]
    width = 2  # means layer --> 2 (3 children per parent)
    low_dim = 4
    rng = npr.default_rng(42)
    LFLR = layers.LayerFactoryFB(num_base_models, models_per_base, xshape, width, low_dim, rng)
    F, _, _ = forest.build_forest(depth, LFLR, 0., 0.)  # no reg
    return F, LFLR, depth


def _get_num_nodes(depth: int, layer_width: int):
    nzs = [(layer_width + 1)**de for de in range(depth)]
    return sum(nzs)


def test_forest_build():
    F, LFLR, depth = _build_forest()
    width = LFLR.get_width()
    assert(width + 1 == len(F.root_node.children))
    for child in F.root_node.children:
        assert(width + 1 == len(child.children))
    # look at all the layers
    w_list = []
    _get_layer_ws(F.root_node, w_list)
    num_exp = _get_num_nodes(depth, width)
    assert(num_exp == len(w_list))
    # makes sure that all ws are different:
    for i in range(len(w_list)):
        ar_i = w_list[i].numpy()
        for j in range(len(w_list)):
            if i == j:
                continue
            ar_j = w_list[j].numpy()
            assert(np.sum(ar_i != ar_j) >= 1)


def test_eval_forest():
    batch_size = 16
    F, LFLR, depth = _build_forest()
    # layer shape?
    sample_layer = LFLR.build_layer()
    w_shape = np.shape(sample_layer._get_wop().numpy())[1:]
    # assuming flattened (which should _build_forest is)
    [num_models, _layer_width, dims] = w_shape
    x = tf.ones([batch_size, dims])
    prs = F.eval(x).numpy()
    num_leaves = (LFLR.get_width() + 1) ** depth
    assert(np.shape(prs) == (batch_size, num_models, num_leaves))
    # check normalization:
    assert(np.all(np.fabs(np.sum(prs, axis=-1) - 1.) < .00001))

    # 0 check: constant within model
    # ... models will be different due to different offsets in layers
    x = tf.zeros([batch_size, dims])
    prs = F.eval(x).numpy()
    prs_mod_red = np.sum(prs, axis=1)
    for i in range(np.shape(prs_mod_red)[0]):
        assert(np.sum(prs[i] - prs[0]) < .00001)

    # HI check --> full offset dim should disapear
    x = tf.ones([batch_size, dims]) * 1000000
    prs = F.eval(x).numpy()
    assert(np.sum(prs[:,:,-1]) < .01)


def test_gauss():
    rng = npr.default_rng(42)
    num_model = 3
    num_state = 4
    dim = 5
    GF = decoders.GaussFull(num_model, num_state, dim, rng)
    # testing of built constructs
    munp = GF.mu.numpy()
    L, Dexp, _ld, _L_trans = GF._get_LDL_comps()
    Lnp = L.numpy()
    Dnp = Dexp.numpy()
    LDLnp = GF._get_LDL().numpy()  # precision matrix
    assert(np.shape(Lnp) == (num_model, num_state, dim, dim))
    assert(np.shape(Dnp) == (num_model, num_state, dim, dim))
    assert(np.shape(LDLnp) == (1, num_model, num_state, dim, dim))  # pad for batch
    tolerance = 1e-6
    for i in range(num_model):
        for j in range(num_state):
            ## L
            assert(np.sum(np.diag(Lnp[i,j]) - 1.) < tolerance)
            assert(np.sum(Lnp[i,j] - np.tril(Lnp[i,j])) < tolerance)
            ## D
            assert(np.sum(Dnp[i,j] - np.tril(Dnp[i,j])) < tolerance)
            assert(np.sum(Dnp[i,j] - np.triu(Dnp[i,j])) < tolerance)
            ## LDL
            # symmetry
            assert(np.sum(LDLnp[0,i,j] - LDLnp[0,i,j].T) < tolerance)
            # determinant: prod of eigenvals = prod of diag(Dnp)
            assert((npla.det(LDLnp[0,i,j]) - np.prod(np.diag(Dnp[i,j]))) < tolerance)

    # log probability simple tests:
    batch_size = 6
    xnp = rng.random((batch_size, dim)).astype(np.float32)
    x = tf.constant(xnp)
    log_prob = GF.calc_log_prob(x)
    lpnp = log_prob.numpy()
    assert(np.shape(log_prob.numpy()) == (batch_size, num_model, num_state))
    #print(log_prob)
    tolerance = 1e-4  # there's gonna be some error in this computation
    for i in range(num_model):
        for j in range(num_state):
            prob = m_normal.pdf(xnp, mean=munp[i,j], cov=npla.pinv(LDLnp[0,i,j]))
            assert(np.sum(lpnp[:,i,j] - np.log(prob)) < tolerance)

    # trainable vars:
    assert(len(GF.get_trainable_weights()) == 3)


def test_GMMForestEM():
    tol = 1e-4
    batch_size = 24
    depth = 3
    base_models = 2
    models_per_base = 4
    num_state = base_models * models_per_base
    xshape = [6,7]
    x = tf.ones([batch_size] + xshape)
    width = 2
    fb_dim = 5
    rng = npr.default_rng(42)
    # single x layer
    layer_factory = layers.LayerFactoryFB(base_models, models_per_base, xshape,
                                            width, fb_dim, rng)

    depth = 3
    gauss_dim = 5
    model = GMMforestEM(depth, layer_factory, gauss_dim, 1., 1., rng)
    forest_eval = model.soft_forest.eval(x)
    assert(np.shape(forest_eval.numpy()) == (batch_size, base_models * models_per_base,
                                                (width+1)**depth))
    
    
    data_weights = tf.ones([batch_size, base_models * models_per_base])
    floss = forest_loss(forest_eval, data_weights)
    assert(np.all(floss.numpy() < 0))  # negentropy should be negative for this case
    assert(np.fabs(np.sum(floss.numpy()) - -413.81592) < tol)
    assert(np.fabs(np.sum(spread_loss(model.ref_layers).numpy()) - 1289.4651) < tol)

    # mixture coefficients/probs
    mix_coeffs = model._get_mixture_prob()
    assert(np.shape(mix_coeffs.numpy()) == (base_models * models_per_base, (width+1)**depth))
    assert(np.all(mix_coeffs.numpy() >= 0.0))
    # norm property
    assert(np.all(np.fabs(np.sum(mix_coeffs.numpy(), axis=-1) - 1.) < tol))

    y = tf.ones([batch_size, gauss_dim])

    # latent probs:
    latent_probs = model.latent_posterior(x, y)
    assert(np.shape(latent_probs.numpy()) == (batch_size, base_models * models_per_base, (width+1)**depth))
    assert(np.all(np.fabs(np.sum(latent_probs.numpy(), axis=-1) - 1.) < tol))

    # loss samples:
    loss_samples, _ = model._loss_samples_noreg(x, y, latent_probs)
    assert(np.shape(loss_samples.numpy()) == (batch_size, base_models * models_per_base, (width+1)**depth))


def _em_helper(model, optimizer, x, y, data_weights, num_epoch=50, num_step=100):
    # EM
    z = None
    losses = []
    mus = [model.decoder.mu.numpy()]
    for _epoch in range(num_epoch):
        z = model.latent_posterior(x, y)
        for _step in range(num_step):
            with tf.GradientTape() as tape:
                loss = model.loss(x, y, data_weights, z)
                grads = tape.gradient(loss, model.get_trainable_weights())
                optimizer.apply_gradients(zip(grads, model.get_trainable_weights()))
        losses.append(model.loss(x, y, data_weights, z))
        mus.append(model.decoder.mu.numpy())
    return np.array(mus), losses


def test_GMMForestEM_simplefit():
    # fit to 2 gaussians
    import pylab as plt
    from tensorflow.keras.optimizers import Adam

    # model generation
    depth = 1  # just the root node
    base_models = 1
    models_per_base = 1
    xshape = [3]  # x, y, constant
    width = 1  # will yield 2 separate branches / states
    fb_dim = 5
    rng = npr.default_rng(42)
    # single x layer
    layer_factory = layers.LayerFactoryFB(base_models, models_per_base, xshape,
                                            width, fb_dim, rng)


    gauss_dim = 2
    model = GMMforestEM(depth, layer_factory, gauss_dim, 10., 0., rng)

    # 2 clusters with no input info
    # Each state should split
    y1 = npr.rand(10, 2) + np.array([2,0])
    y2 = npr.rand(10, 2) + np.array([0,2])
    y = np.vstack([y1, y2]).astype(np.float32)
    x = np.hstack((0 * y, np.ones((20,1)))).astype(np.float32)
    data_weights = np.ones((20,1)).astype(np.float32)

    # EM
    optimizer = Adam(0.1)
    mus, losses = _em_helper(model, optimizer, x, y, data_weights, num_epoch=20)
    
    # mus = tsteps x num_model x num_state x dim

    plt.figure()
    plt.plot(losses)

    plt.figure()
    plt.scatter(y[:,0], y[:,1])
    # model 
    plt.plot(mus[:,0,0,0], mus[:,0,0,1], color='b')
    plt.plot(mus[:,0,1,0], mus[:,0,1,1], color='r')


    # repeat but deterministic
    # should get lower error!
    # for 1 model -->
    # error for x = y*0: -100 (10 epochs); -105 (20 epochs)
    # error for x = y: -112 (10 epochs); -101 (20 epochs lol); -121 (100 epochs)
    # ... lol EM: yeah, super messy but def better

    model = GMMforestEM(depth, layer_factory, gauss_dim, 10., 0., rng)
    x = np.hstack((1 * y, np.ones((20,1)))).astype(np.float32)
    # EM
    optimizer = Adam(0.1)
    mus, losses = _em_helper(model, optimizer, x, y, data_weights, num_epoch=100)

    print('soft forest output?')
    print(model.soft_forest.eval(x))
    
    plt.figure()
    plt.plot(losses)

    plt.figure()
    plt.scatter(y[:,0], y[:,1])
    plt.plot(mus[:,0,0,0], mus[:,0,0,1], color='b')
    plt.plot(mus[:,0,1,0], mus[:,0,1,1], color='r')
    plt.show()


def test_binaryloss_objfunc():
    tol = 1e-4
    # binary loss
    ar_np = np.ones((4, 2)) * -5.
    # class predictions = 0, 0, 1, 1 for both models
    ar_np[2:,:] = 5.
    # truths = 0, 1, 0, 1 (ends correct --> low error end, high error mids)
    truths_np = np.array([0, 1, 0, 1])
    preds = tf.constant(ar_np, tf.float32)
    truths = tf.constant(truths_np, tf.float32)
    BL = BinaryLoss(2)
    bl_loss = BL.loss_sample(preds, truths)
    assert(np.shape(bl_loss.numpy()) == (4,2))
    targ = np.array([[.0067, .0067], [5.0067, 5.0067], [5.0067, 5.0067], [.0067, .0067]])
    assert(np.all((bl_loss.numpy() - targ) < tol))

    # compare to non-expanded formulation
    pred_prob = tf.nn.sigmoid(preds)
    truths_re = tf.reshape(truths, [-1,1])
    loss_ref = -1 * (truths_re * tf.math.log(pred_prob) + (1 - truths_re) * tf.math.log(1 - pred_prob))
    assert(np.all((bl_loss - loss_ref).numpy() < tol))


def test_multiloss_objfunc():
    tol = 1e-4
    # predictions ~ 4 classes (3 batches, 2 parallel models)
    ar_np = np.ones((4, 2, 4))
    # each model predicts 0,1,2,3 in order
    ar_np[0,:,0] = 10
    ar_np[1,:,1] = 10
    ar_np[2,:,2] = 10
    ar_np[3,:,3] = 10
    preds = tf.constant(ar_np, dtype=tf.float32)
    # truths: first 1st and last correct
    # = [0, 0, 3, 3]
    truths_np = np.zeros((4,))
    truths_np[2:] = 3
    truths = tf.constant(truths_np, dtype=tf.int32)
    ML = MultinomialLoss(2)
    ml_loss = ML.loss_sample(preds, truths)
    assert(np.shape(ml_loss.numpy()) == (4, 2))
    targ = np.array([[3.7002563e-04, 3.7002563e-04], [9.0003700e+00, 9.0003700e+00],
                     [9.0003700e+00, 9.0003700e+00], [3.7002563e-04, 3.7002563e-04]])
    assert(np.all((ml_loss.numpy() - targ) < tol))

    # compare to non-expanded formulation
    pred_prob = tf.nn.softmax(preds, axis=-1)
    truths_onehot = np.zeros((4, 4))
    truths_onehot[:2, 0] = 1
    truths_onehot[2:, -1] = 1
    truths_onehot_tf = tf.reshape(tf.constant(truths_onehot, tf.float32), [4, 1, 4])
    ref_err = -1 * tf.reduce_sum(truths_onehot_tf * tf.math.log(pred_prob), axis=-1)
    assert(np.all((ml_loss - ref_err) < tol))


def test_quantileloss_objfunc():
    tol = 1e-4
    # quantile loss testing
    preds_np = np.ones((5, 2, 3))
    preds = tf.constant(preds_np, tf.float32)
    truths = tf.constant(np.linspace(0.0, 2.0, 5), tf.float32)
    taus = tf.constant([.1, .5, .9])
    QL = QuantileLoss(3, 2, taus)
    ql_loss = QL.loss_sample(preds, truths)
    assert(np.shape(ql_loss.numpy()) == (5, 2, 3))
    targ = [[[0.9, 0.5, 0.10000002],
             [0.9, 0.5, 0.10000002]],
            [[0.45, 0.25, 0.05000001],
             [0.45, 0.25, 0.05000001]],
            [[0.,  0.,  0., ],
             [0.,  0.,  0., ]],
            [[0.05, 0.25, 0.45],
            [0.05, 0.25, 0.45]],
            [[0.1, 0.5, 0.9, ],
            [0.1, 0.5, 0.9, ]]]
    assert(np.all((ql_loss - targ) < tol))

    # more easily interpretable test?
    # ensure loss is minimized for correct quantile!
    truths = np.arange(0, 100)  # quantile numbs = 10, 50, 90
    best_preds = np.array([[[10, 50, 90]]])
    best_preds = np.tile(best_preds, (100, 1, 1))
    for scale_direction in [2, -2]:
        lozz = []
        for i in range(5):
            mut_preds = best_preds + scale_direction * i * np.ones((100, 1, 1))  # shifting preds test
            mut_loss = tf.reduce_sum(QL.loss_sample(tf.constant(mut_preds, tf.float32),
                                                    tf.constant(truths, tf.float32)))
            lozz.append(mut_loss.numpy())
        assert(np.all(np.diff(np.array(lozz)) >= 0))



if __name__ == '__main__':
    test_layers()
    test_forest_build()
    test_eval_forest()
    test_gauss()
    #test_GMMForestEM()
    #test_GMMForestEM_simplefit()
    test_binaryloss_objfunc()
    test_multiloss_objfunc()
    test_quantileloss_objfunc()
