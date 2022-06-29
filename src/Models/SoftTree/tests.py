"""Testing forest components"""
from typing import List
from Models.SoftTree import layers
from Models.SoftTree import forest
from Models.SoftTree import decoders
from Models.SoftTree.assembled_models_em import GMMforestEM
from Models.SoftTree.objective_funcs import forest_loss, spread_loss
import Models.SoftTree.train_model as train_model
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
    F, _, _ = forest.build_forest(depth, LFLR)
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
    num_mix = 6
    model = GMMforestEM(depth, layer_factory, num_mix, gauss_dim, 1., 1., rng)
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
    assert(np.shape(mix_coeffs.numpy()) == (base_models * models_per_base, (width+1)**depth, num_mix))
    assert(np.all(mix_coeffs.numpy() >= 0.0))
    # norm property
    assert(np.all(np.fabs(np.sum(mix_coeffs.numpy(), axis=-1) - 1.) < tol))

    # forward probabilties
    # --> batch_size x num_model x num_state x num_mix
    y = tf.ones([batch_size, gauss_dim])
    for_probs = model._forward_probabilities(y)
    assert(np.shape(for_probs.numpy()) == (batch_size, base_models * models_per_base, (width+1)**depth, num_mix))

    # posterior probabilities:
    post_probs = model._posterior_probabilities(for_probs)
    assert(np.shape(for_probs.numpy()) == np.shape(post_probs.numpy()))
    assert(np.all(np.fabs(np.sum(post_probs.numpy(), axis=-1) - 1.) < tol))

    # latent probs:
    latent_probs = model.latent_posterior(x, y)
    assert(np.shape(for_probs.numpy()) == np.shape(latent_probs.numpy()))
    assert(np.all(np.fabs(np.sum(latent_probs.numpy(), axis=-1) - 1.) < tol))

    # loss samples:
    loss_samples, _ = model._loss_samples_noreg(x, y, latent_probs)
    assert(np.shape(loss_samples.numpy()) == (batch_size, base_models * models_per_base, (width+1)**depth, num_mix))


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
    xshape = [2]
    width = 1  # will yield 2 separate branches / states
    fb_dim = 5
    rng = npr.default_rng(42)
    # single x layer
    layer_factory = layers.LayerFactoryFB(base_models, models_per_base, xshape,
                                            width, fb_dim, rng)


    gauss_dim = 2
    num_mix = 2
    model = GMMforestEM(depth, layer_factory, num_mix, gauss_dim, 10., 0., rng)

    # 2 clusters with no input info
    # Each state should split
    y1 = npr.rand(10, 2) + np.array([2,0])
    y2 = npr.rand(10, 2) + np.array([0,2])
    y = np.vstack([y1, y2]).astype(np.float32)
    x = (0. * y).astype(np.float32)  # removes information
    data_weights = np.ones((10,1)).astype(np.float32)


    # EM
    optimizer = Adam(0.1)
    mus, losses = _em_helper(model, optimizer, x, y, data_weights, num_epoch=20)
    
    plt.figure()
    plt.plot(losses)

    plt.figure()
    plt.scatter(y[:,0], y[:,1])
    # blue = state 1
    plt.plot(mus[:,0,0,0], mus[:,0,0,1], color='b')
    plt.plot(mus[:,0,1,0], mus[:,0,1,1], color='b')
    # red = state 2
    plt.plot(mus[:,0,2,0], mus[:,0,2,1], color='r')
    plt.plot(mus[:,0,3,0], mus[:,0,3,1], color='r')

    plt.show()


    # repeat but deterministic
    # should get state separation
    # --> current finding = crappy sub/mix states die --> collapses to deterministic
    # TODO: this one's having some issues now ~ one of the states is collapsing
    # TODO: is it because initial estimate is trash?
    # TODO: is it because we upped the number of steps per epoch?

    # TODO: TEMPERATURE PLAN
    # > limit the range of probabilities for latent states (early: max = 0.8 or something)
    # > as iter, increase the range --> allow approx hard assignment

    # OBSERVATION: low forest penalty --> one branch dies
    # HIGH FOREST PENALTY --> chaotic results
    model = GMMforestEM(depth, layer_factory, num_mix, gauss_dim, 0., 0., rng)
    # TODO/TESTING: trying so that means are centered
    y1 = npr.rand(100, 2) + np.array([2,0])  # TODO: try different data set sizes
    y2 = npr.rand(100, 2) + np.array([0,2])
    y = np.vstack([y1, y2]).astype(np.float32)
    x = y
    optimizer = Adam(0.1)
    mus, losses = _em_helper(model, optimizer, x, y, data_weights, num_epoch=100, num_step=10)

    plt.figure()
    plt.plot(losses)

    plt.figure()
    plt.scatter(y[:,0], y[:,1])
    # blue = state 1
    plt.plot(mus[:,0,0,0], mus[:,0,0,1], color='b')
    plt.plot(mus[:,0,1,0], mus[:,0,1,1], color='b')
    # red = state 2
    plt.plot(mus[:,0,2,0], mus[:,0,2,1], color='r')
    plt.plot(mus[:,0,3,0], mus[:,0,3,1], color='r')

    plt.show()


if __name__ == '__main__':
    test_layers()
    test_forest_build()
    test_eval_forest()
    test_gauss()
    test_GMMForestEM()
    test_GMMForestEM_simplefit()
