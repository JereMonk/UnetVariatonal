import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
  def __init__(self,input_size,do_dropout=False,l1_reg=0,l2_reg=0):
        
        super(Encoder, self).__init__(self)
        
        self.do_dropout=do_dropout
   
  
        self.input_size=input_size
       
        
        self.inputs = tf.keras.Input(shape=(input_size,input_size,3))
        


        self.conv1 = tf.keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=(2,2),padding='same',name='CONV11',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN1 = tf.keras.layers.BatchNormalization(name='BN11')
        self.ReLU1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=(2,2),padding='same',name='CONV21',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN2 = tf.keras.layers.BatchNormalization(name='BN21')
        self.ReLU2 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding='same',name='CONV22',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN3 = tf.keras.layers.BatchNormalization(name='BN22')
        self.ReLU3 = tf.keras.layers.ReLU()

       
        self.conv4 = tf.keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding='same',name='CONV31',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN4 = tf.keras.layers.BatchNormalization(name='BN31')
        self.ReLU4 = tf.keras.layers.ReLU()

        self.conv5 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(4,4),strides=(2,2),padding='same',name='CONV32',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN5 = tf.keras.layers.BatchNormalization(name='BN32')
        self.ReLU5 = tf.keras.layers.ReLU()
        
        self.flatten=tf.keras.layers.Flatten()
        self.mu = tf.keras.layers.Dense(units=1024)
        self.sigma = tf.keras.layers.Dense(units=1024)

        self.sampler = Sampling()

        
    
  def call(self,x,training=False):

      
      x1= self.conv1(x)
      x1= self.BN1(x1)
      x1=self.ReLU1(x1)
      
      x2= self.conv2(x1)
      x2= self.BN2(x2)
      x2=self.ReLU2(x2)
      

      x3= self.conv3(x2)
      x3= self.BN3(x3)
      x3=self.ReLU3(x3)
   

      x4= self.conv4(x3)
      x4= self.BN4(x4)
      x4=self.ReLU4(x4)

      x5= self.conv5(x4)
      x5= self.BN5(x5)
      x5=self.ReLU5(x5)

      x6 = self.flatten(x5)

      mean = self.mu(x6)
      var = self.sigma(x6)
      sampled = self.sampler((mean,var))

      return(mean,var,sampled)

class Decoder(tf.keras.Model):

  def __init__(self,do_dropout=False,l1_reg=0,l2_reg=0):
        super(Decoder, self).__init__()
        
        self.do_dropout=do_dropout

        self.dense= tf.keras.layers.Dense(4 * 4 * 1024, activation="relu")
        self.reshape = tf.keras.layers.Reshape((4, 4, 1024))

        self.upsample1 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),padding='same',name='UPSAMPLE1',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.ReLU1 = tf.keras.layers.ReLU()

        self.upsample2 = tf.keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),padding='same',name='UPSAMPLE2',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN2 = tf.keras.layers.BatchNormalization()
        self.ReLU2 = tf.keras.layers.ReLU()

        self.upsample3 = tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),strides=(2,2),padding='same',name='UPSAMPLE3',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN3 = tf.keras.layers.BatchNormalization()
        self.ReLU3 = tf.keras.layers.ReLU()

        self.upsample4 = tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),strides=(2,2),padding='same',name='UPSAMPLE4',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN4 = tf.keras.layers.BatchNormalization()
        self.ReLU4 = tf.keras.layers.ReLU()

        self.upsample5 = tf.keras.layers.Conv2DTranspose(filters=3,kernel_size=(4,4),strides=(2,2),padding='same',name='UPSAMPLE4',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg))
        self.BN5 = tf.keras.layers.BatchNormalization()
        self.ReLU5 = tf.keras.layers.ReLU()

    
        self.last_acti = tf.keras.layers.Activation('sigmoid')
        
    
  def call(self,x,training=False):
      
      x = self.dense(x)
      x = self.reshape(x)
      
      
      
      x1= self.upsample1(x)
      x1= self.BN1(x1)
      x1=self.ReLU1(x1)

      x2= self.upsample2(x1)
      x2= self.BN2(x2)
      x2=self.ReLU2(x2)

      x3= self.upsample3(x2)
      x3= self.BN3(x3)
      x3=self.ReLU3(x3)

      x4= self.upsample4(x3)
      x4= self.BN4(x4)
      x4=self.ReLU4(x4)

      x5= self.upsample5(x4)
      x5= self.BN5(x5)
      x5=self.ReLU5(x5)



      output = self.last_acti(x5)

      
      return(output)

      
class VariatonalAutoEncoder(tf.keras.Model):
  def __init__(self,encoder,decoder):
    super(VariatonalAutoEncoder, self).__init__()
    self.encoder=encoder
    self.decoder=decoder

  def call(self,x):
    mean,var,sampled = self.encoder(x)
    z = self.decoder(sampled)
    return(z)

  def compile(
      self,
      optimizer,
      loss_reco,
  
    ):
    super(VariatonalAutoEncoder, self).compile()

    self.optimizer = optimizer
    self.loss_reco = loss_reco
    

  @tf.function
  def train_step(self,batch_data):

    with tf.GradientTape() as tape:

      mean,var,sampled = self.encoder(batch_data)
      z = self.decoder(sampled)      
      loss_reco = self.loss_reco(batch_data,z)

      kl_loss = -0.5 * (1 + var - tf.square(mean) - tf.exp(var))
      kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
      total_loss = loss_reco + kl_loss
      
      for layer in self.layers:
        total_loss+=tf.math.reduce_sum(layer.losses)

    grads = tape.gradient(total_loss, self.trainable_variables)
      
    self.optimizer.apply_gradients(
        zip(grads, self.trainable_variables)
    )

    return({
      "kl_loss":kl_loss,
      "loss_reco":loss_reco,
      "total_loss":total_loss
    })
