import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from Model import Model
import Plots

def assert_column(x):
    assert x.ndim == 2
    assert x.shape[1] == 1

def sigmoid(x):
    assert_column(x)
    den = 1. + np.exp(-x)
    num = 1.      
    return np.divide(num, den)

def sigmoid_d(x, y):
    return np.multiply(y, 1-y)

def tanh(x):
    y = np.tanh(np.array(x,dtype=float))
    return y

def tanh_d(x, y):
    for i in range(len(x)):
        assert np.tanh(x[i].astype(float)) == y[i]


            
    return 1 - np.square(y)

def relu(x):
    return np.maximum(0,x)

def relu_d(x):
    dx = np.vectorize(lambda z: 1 if z > 0 else 0)
    return dx(x)

def softmax_prob(x):
    assert_column(x)
    p_raw = np.exp(np.array(x,dtype=float))
    p_sum = np.sum(p_raw)
    return np.divide(p_raw, p_sum)

def softmax_cost(inputs, y):
    """ admits a column vector of inputs, 
        and a (1,) numpmy array y indicating the class
    """
    assert_column(inputs)
    probs = softmax_prob(inputs)
    return -np.log(probs[y[0]])[0]

def softmax_cost_d(inputs, y_vec):
    return softmax_prob(inputs) - y_vec

def reg_l1_d(W, eps=0.0001):
    W_sq = np.square(W)
    W_sq_sum = np.sum(W_sq)
    coef = pow(W_sq_sum + eps, -0.5) 
    return coef * W

def reg_l1(W, eps=0.0001):
    W_sq = np.square(W)
    W_sq_sum = np.sum(W_sq)
    return pow(W_sq_sum + eps, 0.5)

def reg_l2_d(W):
    return 2*W

def reg_l2(W):
    return np.sum(np.square(W))


class NeuralNetwork(Model):
    def __init__(self, x_mat, y_col, act="tanh", units=[2]):
        Model.__init__(self)
        self.x_mat = x_mat
        self.y_col = y_col
        self.act_f, self.act_b = self._get_act_functions(act)
        self.nl = len(units) + 1
        self.units = [self.x_mat.shape[1]] + units
   
        self.wts = self._init_wts()
        self.loss = []

    """
    Schematic for nl = 3:
        - 1 input layer
        - 1 hidden layer
        - 1 output layer

        L1          L2z       L2a         L3z      L3a

      act[0]      inp[1]    act[1]      inp[2]    act[2]
      -------     -------   -------     -------   -------   -----
      |input|     |layer|   |acti-|     |final|   |acti-|   |   |
      |layer|-----|  2  |---|vated|-----|layer|---|vated|---| J |
      | :a1 |  |  | :z2 | | | :a2 |  |  | :z3 | | | :a3 |   |   |
      -------  |  ------- | -------  |  ------- | -------   -----
               |          |          |          |
            -------    -------    -------    -------
            |f(a1)|    |s(z2)|    |f(a2)|    |s(z3)|
            | :W1 |    -------    | :W2 |    | =z3 |
            | :b1 |               | :b2 |    -------
            -------               -------
            wts[0]                wts[1]

    """
    def compute_loss(self):
        m = self.x_mat.shape[0]
        c = 0
        for i in range(m):
            x_col = self.x_mat[i].reshape((self.x_mat.shape[1],1))
            y = self.y_col[i]
            output = self._feedforward(x_col)[1][-1]
            cost_i = softmax_cost(output, y)
            c += cost_i
        #print "Entropy Loss:", c
        if self.reg_method == 1:
            for i in range(self.nl-1):
                cost_r = self.l1 * reg_l1(self.wts[i][0], self.eps)   
                c += cost_r
        elif self.reg_method == 2:
            for i in range(self.nl-1):
                cost_r = self.l2 * reg_l2(self.wts[i][0]) 
                c += cost_r
        else: 
            for i in range(self.nl-1):
                cost_r = self.l2 * reg_l2(self.wts[i][0])
                if i == 0:
                    #cost_r += self.l1 * reg_l1(self.wts[i][0], self.eps)  
                    cost_r = self.l1 * reg_l1(self.wts[i][0], self.eps)   
                c += cost_r
        return c

    def print_wts(self):
        for i in range(len(self.wts)):
            W, b = self.wts[i]
            print str("W"+str(i))
            print W
            print str("b"+str(i))
            print b

    def feature_impact(self):
        W = self.wts[0][0]
        imp = np.sum(np.square(W), axis=0)
        return imp

    def train_error(self):
        preds = self.predict(self.x_mat)
        correct = [pred == ans for pred,ans in zip(preds, np.squeeze(self.y_col.tolist()))]
        return  1- sum(correct)/float(len(correct))
       
    def predict_one(self, arr):
        arr = np.array(arr).reshape((1,len(arr)))
        return self.predict(arr)[0]

    def predict(self, data_rows):
        if data_rows.ndim == 1: # an array of features is passed in
            data_rows = np.array(data_rows).reshape((1,len(data_rows)))
        prediction = [None for _ in xrange(data_rows.shape[0])]
        n_feat = data_rows.shape[1]
        for i in xrange(data_rows.shape[0]):
            x_col = data_rows[i].reshape((n_feat,1))
            output = self._feedforward(x_col)[1][-1]
            probs = softmax_prob(output)
            max_prob = max(probs)[0]
            prediction[i] = np.where(probs == max_prob)[0][0]
        return prediction

    def _compare_loss(self, prev_loss):
        curr_loss = self.compute_loss()
        if prev_loss is None:
            return (curr_loss, 0)
        else:
            delta = curr_loss - prev_loss
            delta_percent = delta / prev_loss
            return (curr_loss, delta_percent)

    def train(self, max_iter=10000, a=5e-2, l2=1e-1, e=1e-5, t=1e-4,\
              v=1, l1=0.5, method="minibatch", reg_method=3):
        self.alp, self.l2, self.eps = a, l2, e
        self.l1=l1
        self.reg_method = reg_method
        method_map = {"minibatch":self._minibatch_gd, "batch":self._batch_gd}

        fail_multiplier = 1
        record_sparsity = 100
        self.loss = [None] * (max_iter/record_sparsity + 1)
        prev_loss = None        
        for i in xrange(max_iter):
            method_map[method]()
            if i % (10*record_sparsity) == 0:
                self.alp *= 0.85
            if i % record_sparsity == 0:
                prev_loss, delta = self._compare_loss(prev_loss)
                self.loss[i/record_sparsity] = prev_loss
                if delta > (fail_multiplier * t):
                    print "gradient diverged"
                    raise ValueError("Gradient Descent Diverged")
                elif delta < 0 and abs(delta) <= t:            
                    last = i/record_sparsity
                    self.loss = self.loss[0:last]
                    break
            if v > 0 and (i % (record_sparsity * v) == 0 or i == max_iter-1):
                if i == max_iter-1:    
                    prev_loss, delta = self._compare_loss(prev_loss)
                self._print_progress(i, prev_loss, delta)

    def _print_progress(self, i, loss, delta):
        if i == 0:
            if self.reg_method == 1:
                print "L1", self.l1
            elif self.reg_method == 2:
                print "L2", self.l2
            else:
                print "L2", self.l2, "L1", self.l1
            print "         Iter | train loss |   delta     | alpha"
        print "        {0:05d} | {1:10.6f} | {2:.4e} | {3:.4e}"\
              .format(i, loss, delta, self.alp)

    def _batch_gd(self):
        agg_grads = self._init_agg_delta()
        for i in xrange(self.x_mat.shape[0]):
            x_col = self.x_mat[i].reshape((self.x_mat.shape[1],1))
            grads = self._forward_backward(x_col, self.y_col[i])
            self._update_agg_grads(agg_grads, grads)
        self._average_agg_grads(agg_grads)
        self._add_reg_grads(agg_grads)
        self._gradient_update(agg_grads)

    def _minibatch_gd(self, nb=5):
        nrows = self.x_mat.shape[0]
        row_order = np.arange(nrows)
        np.random.shuffle(row_order)
        if nrows < nb * 2:
            nb = 1
        for batch in xrange(nb):
            agg_grads = self._init_agg_delta()
            s_pos = int(math.ceil(nrows/float(nb)) * batch)
            e_pos = min(int(math.ceil(nrows/float(nb)) * (batch+1)), nrows)
            for i in xrange(s_pos,e_pos):
                x_col = self.x_mat[row_order[i]].reshape((self.x_mat.shape[1],1))
                grads = self._forward_backward(x_col, self.y_col[row_order[i]])
                self._update_agg_grads(agg_grads, grads)
            self._average_agg_grads(agg_grads, e_pos-s_pos)
            self._add_reg_grads(agg_grads)
            self._gradient_update(agg_grads)

    def _forward_backward(self, x, y):
        assert_column(x)
        inp_layer, act_layer = self._feedforward(x)
        cost = softmax_cost(act_layer[-1], y)
        deltas = self._backprop(inp_layer, act_layer, cost, y)
        grads = self._compute_gradients(deltas, act_layer)
        return grads

    def _backprop(self, inp_layer, act_layer, cost, y):
        deltas = [None for _ in xrange(self.nl)]
        y_vec = self._vectorize_y(y)
        probs = softmax_prob(act_layer[-1]) 
        deltas[-1] = probs - y_vec
        for i in sorted(range(1,self.nl-1),reverse=True):
            W_i, b_i = self.wts[i]
            inp = inp_layer[i]
            out = act_layer[i]
            deltas[i] = np.multiply(np.dot(W_i.T,deltas[i+1]),self.act_b(inp,out))
        return deltas

    def _compute_gradients(self, deltas, act_layer):
        grads = [None for _ in xrange(self.nl-1)]
        for i in reversed(range(self.nl-1)):
            dW_i = deltas[i+1].dot(act_layer[i].T)
            db_i = deltas[i+1]
            grads[i] = [dW_i, db_i]
        return grads

    def _update_agg_grads(self, aggs, grads):
        for i in range(len(aggs)):
            Wg, bg = grads[i]
            aggs[i][0] = aggs[i][0] + Wg
            aggs[i][1] = aggs[i][1] + bg
        
    def _gradient_update(self, aggs):
        for i in xrange(len(aggs)):
            W_a, b_a = aggs[i]
            # W_i update
            self.wts[i][0] = self.wts[i][0] - self.alp * W_a
            # b_i update
            self.wts[i][1] = self.wts[i][1] - self.alp * b_a

    def _average_agg_grads(self, aggs, batch_size=None):
        if batch_size is None:
            m = self.x_mat.shape[0]
        else:
            m = batch_size
        for i in range(len(aggs)):
            aggs[i][0] = np.divide(aggs[i][0],m)
            aggs[i][1] = np.divide(aggs[i][1],m)

    def _add_reg_grads(self, aggs):
        if self.reg_method == 1:
            for i in xrange(len(aggs)):
                dW_a, db_a = aggs[i]
                W, b = self.wts[i]
                reg_d = self.l1*reg_l1_d(W)
                aggs[i][0] = dW_a + reg_d
        elif self.reg_method == 2:
            for i in xrange(len(aggs)):
                dW_a, db_a = aggs[i]
                W, b = self.wts[i]
                reg_d = self.l2*reg_l2_d(W)
                aggs[i][0] = dW_a + reg_d
        else:
            for i in xrange(len(aggs)):
                dW_a, db_a = aggs[i]
                W, b = self.wts[i]
                reg_d = self.l2*reg_l2_d(W)
                if i == 0:
                    reg_d = self.l1*reg_l1_d(W, self.eps)
                aggs[i][0] = dW_a + reg_d




    def _init_agg_delta(self):
        aggs = [None for _ in range(self.nl-1)]
        for i in xrange(self.nl-1):
            W_agg = np.zeros(self.wts[i][0].shape)
            b_agg = np.zeros(self.wts[i][1].shape)
            aggs[i] = [W_agg, b_agg]            
        return aggs



    def _feedforward(self, x):
        assert_column(x)
        inp_layer = [x] + [None for _ in xrange(1,self.nl)]
        act_layer = [x] + [None for _ in xrange(1,self.nl)]
        for i in xrange(1,self.nl):                
            W_i_minus, b_i_minus = self.wts[i-1]
            inp_layer[i] = np.dot(W_i_minus,act_layer[i-1]) + b_i_minus
            # activate layer if layer is not L_nl
            if i == self.nl-1: 
                act_layer[i] = inp_layer[i]
            else:
                act_layer[i] = self.act_f(inp_layer[i])            
        return (inp_layer, act_layer)

    def _vectorize_y(self, y):
        y_vec = np.insert(np.zeros(self.units[-1]-1), y[0], 1)
        return y_vec.reshape((self.units[-1],1))

             
    def _init_wts(self):    
        wts = [None for _ in range(self.nl-1)]
        for i in xrange(self.nl-1):
            # Xavier's initialization
            W = np.random.randn(self.units[i+1],self.units[i]) * \
                                np.sqrt(2./self.units[i-1])
            b = np.zeros((self.units[i+1],1))
            wts[i] = [W, b]
        return wts
        

    def _get_act_functions(self, s):
        if s == "tanh":
            return tanh, tanh_d
        elif s == "relu":
            return relu, relu_d
        elif s == "sigmoid":
            return sigmoid, sigmoid_d
        else:
            print "function not recognized. Defaults to tanh"
            return tanh, tanh_d

def neural_cv(lamb, xmat, ymat, units, k=10, max_iter=3000, alp=0.05, eps=1e-6, thr=1e-4, reg_method=3, v=1):
    l1, l2 = lamb
    #print " "*7, "Performing {}-Fold CV...".format(k)
    nrows = xmat.shape[0]
    row_order = np.arange(nrows)
    np.random.shuffle(row_order)
    cv_error = []
    train_loss = []
    for batch in xrange(k):
        s_pos = int(math.floor(nrows/float(k)) * batch)
        e_pos = int(math.floor(nrows/float(k)) * (batch+1))
        if batch == k-1: e_pos = nrows
        this_e, this_l = validate_batch(xmat, ymat, s_pos, e_pos, row_order, \
                               max_iter, alp, l2, eps, thr, units, v, l1, reg_method)
        #print "Batch {} Test Error: {:10.6f}".format(batch+1, this_e)
        cv_error.append(this_e)
        train_loss.append(this_l)
    print " "*7, "Mean CV", sum(cv_error)/float(len(cv_error))
    return (cv_error, train_loss)


def cross_validation(xmat, ymat, units, k=10, max_iter=100000, \
                     alp=0.05, lam=0.1, eps=1e-6, thr=1e-4, v=1, l1=1, reg_method=3):
    """ Deprecated version of cross_validation"""
    #print " "*7, "Performing {}-Fold CV...".format(k)
    nrows = xmat.shape[0]
    row_order = np.arange(nrows)
    np.random.shuffle(row_order)
    cv_error = []
    train_loss = []
    for batch in xrange(k):
        s_pos = int(math.floor(nrows/float(k)) * batch)
        e_pos = int(math.floor(nrows/float(k)) * (batch+1))
        if batch == k-1: e_pos = nrows
        this_e, this_l = validate_batch(xmat, ymat, s_pos, e_pos, row_order, \
                               max_iter, alp, lam, eps, thr, units, v, l1, reg_method)
        #print "Batch {} Test Error: {:10.6f}".format(batch+1, this_e)
        cv_error.append(this_e)
        train_loss.append(this_l)
    print " "*7, "Mean CV", sum(cv_error)/float(len(cv_error)), \
          "at L1 {0:4.3f}, L2 {1:4.3f}".format(l1, lam)
    return (cv_error, train_loss)
    

def validate_batch(xmat, ymat, s_p, e_p, order, \
                   max_iter, alp, lam, eps, thr, units, v, l1, reg_method):

    index_test = order[s_p:e_p]
    x_test = xmat[index_test,:]
    y_test = ymat[index_test,:]

    index_train = [e for e in order if e not in index_test]
    x_train = xmat[index_train,:]
    y_train = ymat[index_train,:]
    
    net = NeuralNetwork(x_train, y_train, units=units)
    net.train(max_iter, alp, lam, eps, thr, v, l1, reg_method=reg_method)
    train_loss = net.loss

    preds = net.predict(x_test)
    correct = [pred == ans for pred,ans in zip(preds, np.squeeze(y_test.tolist()))]
    cv_error = 1- sum(correct)/float(len(correct))

    return (cv_error, train_loss)
    

         









      
        
        
