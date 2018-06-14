import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import lib
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

EPS = 1e-8

class OptimalEncoding(object):
    def __init__(self, encoder, decoder, n_cts, n_categorical):
        self.onehot = OneHotEncoder(n_values=n_categorical, sparse=False)
        
        self.n_cts = n_cts
        self.n_categorical = n_categorical
        
        #Encoder n_out must equal 2*n_cts + n_categorical
        self.encoder = encoder
        #Decoder n_in must equal n_cts + n_categorical
        self.decoder = decoder
        
        self.params = encoder.params + decoder.params
        
        self.X_dim = encoder.n_in
        
        #Data
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.n_cts + self.n_categorical], name='Z')
        
        ####################BACKWARDS (FROM VANILLA VAE)####################################################################
        
        #Encode for reparametrization trick
        self.qZ_params = encoder(self.X, linear_out=True)
        self.qZ_cts_params = self.qZ_params[:, :2*self.n_cts]
        self.qZ_means = self.qZ_cts_params[:,:self.n_cts]
        self.qZ_stds = self.qZ_cts_params[:,-self.n_cts:]
        self.qZ_categorical_params = self.qZ_params[:, 2*self.n_cts:]
        
        #Placeholder for normal samples for reparametrization trick
        self.eps = tf.placeholder(tf.float32, shape=[None, self.n_cts], name='g')
        #Cts reparametrization
        self.qZ_cts = self.qZ_stds*self.eps + self.qZ_means
        
        #Placeholder for gumbel samples for reparametrization trick
        self.g = tf.placeholder(tf.float32, shape=[None, self.n_categorical], name='g')
        #Placeholder for gumbel softmax temperature
        self.tau = tf.placeholder(tf.float32, shape=(), name = 'tau')
        #Gumbel reparametrization
        self.qZ_probs = tf.nn.softmax(self.qZ_categorical_params, axis=-1)
        self.qZ_categorical = lib.gumbel_soft(self.qZ_probs, self.g, tau = self.tau)
        
        #Putting it together
        self.qZ = tf.concat([self.qZ_cts, self.qZ_categorical], axis=1)
        
        #Decode
        self.qX = tf.tanh(decoder(self.qZ, linear_out=True))
        
        #Loss
        #Autoencoder term
        self.Eqxz_logpxz = tf.reduce_mean(tf.reduce_sum(tf.abs(self.X - self.qX), 1))
        #Prior term
        self.KL_qzx_pz_cts = tf.reduce_mean(
            0.5 * tf.reduce_sum(
                tf.square(self.qZ_means) + tf.square(self.qZ_stds) - tf.log(tf.square(self.qZ_stds) + EPS) - 1,
                1
            )
        ) 
        self.KL_qzx_pz_categorical = tf.reduce_mean(
            tf.reduce_sum(
                self.qZ_probs * tf.log(self.qZ_probs / np.float32(self.n_categorical) + EPS),
                1
            )
        )
        self.KL_qzx_pz = self.KL_qzx_pz_cts + self.KL_qzx_pz_categorical
        #Together now
        self.Loss_backwards = self.Eqxz_logpxz + self.KL_qzx_pz
        
        ####################FORWARDS (NEW TO OPTIMAL ENCODING)###############################################################

        #Decode
        self.pX = tf.tanh(decoder(self.Z, linear_out=True))
        
        #Encode for reparametrization trick
        self.pZ_params = encoder(self.pX, linear_out=True)
        self.pZ_cts_params = self.pZ_params[:, :2*self.n_cts]
        self.pZ_means = self.pZ_cts_params[:,:self.n_cts]
        #self.pZ_stds = self.pZ_cts_params[:,-self.n_cts:]
        self.pZ_categorical_params = self.pZ_params[:, 2*self.n_cts:]
        self.pZ_probs = tf.nn.softmax(self.pZ_categorical_params, axis=-1)
        
        #Loss
        self.Epxz_logqzx_cts = tf.reduce_mean(tf.reduce_sum(tf.square(self.Z[:, :self.n_cts] - self.pZ_means), 1))
        self.Epxz_logqzx_categorical = tf.reduce_mean(tf.reduce_mean(-tf.log(self.Z[:, self.n_cts:]*self.pZ_probs + EPS), 1))
        self.Epxz_logqzx = self.Epxz_logqzx_cts + self.Epxz_logqzx_categorical
        self.Loss_forwards = self.Epxz_logqzx
        
    def train(self, x, optimal = True, epochs=100, batch_size=64, lr=1e-3, tau_rate = 1e-4):
        schedule = lambda i: np.float32(np.max((0.5, np.exp(-tau_rate*i))))

        
        ###TOTAL LOSS
        self.Loss = self.Loss_backwards + (self.Loss_forwards if optimal else 0)
        
        #Optimizer 
        solver = tf.train.AdagradOptimizer(learning_rate = lr).minimize(self.Loss, var_list=self.params)

        #Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.sess = sess
        with sess.as_default():
            sess.run(init)

            losses = []
            
            n_batches = int(x.shape[0]/float(batch_size))
            for epoch in tqdm(range(epochs)):
                rand_idxs = np.arange(x.shape[0]) 
                np.random.shuffle(rand_idxs)
                
                loss = 0
                for batch in range(n_batches):
                    tau = schedule(epoch*n_batches + batch)
                    
                    mb_idx = rand_idxs[batch*batch_size:(batch+1)*batch_size]
                    x_mb = x[mb_idx]
                    
                    eps = np.random.normal(size=(batch_size, self.n_cts))
                    g = np.random.gumbel(size=(batch_size, self.n_categorical))
                    
                    z_cts = np.random.normal(size=(batch_size, self.n_cts))
                    if self.n_categorical > 0:
                        z_categorical = self.onehot.fit_transform(np.random.randint(self.n_categorical, size=(batch_size,1)))
                        z = np.hstack((z_cts, z_categorical))
                    else: z = z_cts
                    
                    
                    _, loss_curr = sess.run(
                        [solver, self.Loss], 
                        feed_dict = {self.X:x_mb, self.Z:z, self.eps:eps, self.g: g, self.tau: tau}
                    )
                    
                    loss += loss_curr/n_batches
                    
                losses.append(loss)
            
            plt.figure()
            plt.plot(losses)
            plt.title('total loss')
            

    def encode(self, x, tau = 1e-10):
        eps = np.random.normal(size=(len(x), self.n_cts))
        g = np.random.gumbel(size=(len(x), self.n_categorical))
        return(self.sess.run(self.qZ, feed_dict = {self.X:x, self.eps: eps, self.g: g, self.tau: tau}))
    
    def decode(self, z):
        return(self.sess.run(tf.tanh(self.decoder(z, linear_out=True))))
    
    def sample(self, n):
        z_cts = np.random.normal(size=(n, self.n_cts))
        if self.n_categorical > 0:
            z_categorical = self.onehot.fit_transform(np.random.randint(self.n_categorical, size=(n,1)))
            z = np.hstack((z_cts, z_categorical))
        else: z = z_cts
        return(self.decode(z))