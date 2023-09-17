import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt

class pinns(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(pinns, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.sopt = lbfgs_optimizer(self.trainable_variables)
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.nu = 1/3000
        self.coefs = [0.1, 0.1, 1.0]
    
  
    @tf.function
    def net_f(self, cp):
        x = cp[:, 0]
        y = cp[:, 1]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
          
            X = tf.stack([x, y], axis=1)
            pred = self.model(X)
            pred = self.scale_r(pred)
            U = pred[:, 0] 
            uv = pred[:, 1]
            uu = pred[:, 2]
            vv = pred[:, 3]
            V = pred[:, 4] 
           
            U_x = tape.gradient(U, x)
            U_y = tape.gradient(U, y)
            V_x = tape.gradient(V, x)
            V_y = tape.gradient(V, y)
        U_xx = tape.gradient(U_x, x)
        U_yy = tape.gradient(U_y, y)
        V_xx = tape.gradient(V_x, x)
        V_yy = tape.gradient(V_y, y)
        uv_y = tape.gradient(uv, y)
        uv_x = tape.gradient(uv, x)
        uu_x = tape.gradient(uu, x)
        vv_y = tape.gradient(vv, y)
        
      
        f1 = U * U_x + V * U_y -  self.nu * (U_xx + U_yy) + uu_x + uv_y
        f2 = U * V_x + V * V_y -  self.nu * (V_xx + V_yy) + uv_x + vv_y
        f3 = U_x + V_y
        
        f = tf.stack([f1, f2, f3], axis = -1) * self.coefs

        return f
    
    
    @tf.function
    def train_step(self, bc, cp):
        xy_bc = bc[:, :2]
        u_bc = bc[:, 2:]
        with tf.GradientTape() as tape:
            u_p_bc = self.model(xy_bc)
            
            f = self.net_f(cp)
            
            loss_u = tf.reduce_mean(tf.square(u_bc - u_p_bc[:, :-1]))
            loss_f = tf.reduce_mean(tf.square(f))
            loss = loss_u + loss_f
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads, tf.stack([l1, l2, l3])
    
    def scale(self, y):
        ymax = tf.reduce_max(y, axis = 0)
        ymin = tf.reduce_min(y, axis = 0)
        print(ymax)
        self.ymax = tf.concat([ymax, tf.constant([1.0])], 0)
        self.ymin = tf.concat([ymin, tf.constant([0.0])], 0)
        ys = ((y - ymin) / (ymax - ymin))
        return ys
    
    def scale_r(self, ys):
        y = (ys) * (self.ymax - self.ymin) + self.ymin
        return y
    
    def fit(self, bc, cp):
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        x_bc = bc[:, :2]
        y_bc = bc[:, 2:]
        
        y_bc = self.scale(y_bc)
        
        bc = tf.concat([x_bc, y_bc], axis = 1)
        
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            
            
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        u_p = self.model(cp)
        u_p = self.scale_r(u_p)
        return u_p.numpy()
    
    
class lbfgs_optimizer():
    def __init__(self, trainable_vars, method = 'L-BFGS-B'):
        super(lbfgs_optimizer, self).__init__()
        self.trainable_variables = trainable_vars
        self.method = method
        
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        count = 0
        idx = [] # stitch indices
        part = [] # partition indices
    
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n
    
        self.part = tf.constant(part)
        self.idx = idx
    
    def assign_params(self, params_1d):
        params_1d = tf.cast(params_1d, dtype = tf.float32)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
    def minimize(self, func):
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        results = sopt.minimize(fun = func, 
                            x0 = init_params, 
                            method = self.method,
                            jac = True, options = {'iprint' : 0,
                                                   'maxiter': 50000,
                                                   'maxfun' : 50000,
                                                   'maxcor' : 50,
                                                   'maxls': 50,
                                                   'gtol': 1.0 * np.finfo(float).eps,
                                                   'ftol' : 1.0 * np.finfo(float).eps})