
import tensorflow as tf
from PIL import Image 
import numpy as np
import PIL 


def rescale(image):
    return( np.array(((image))*255 ).astype("uint8") )

def set_learning_rate(step_counter,model,base_lr,steps,decay_step,decay_rate):
    
    if(step_counter<=decay_step):
        new_lr = base_lr
    
    else:
        new_lr = base_lr**(decay_rate*(step_counter//decay_step))
    model.optimizer.lr = new_lr
    
def training(model,train_dataset,max_iter,start_iter,base_lr,ckpt_freq,dir_path,solver_steps,decay_step,decay_rate):
  
    ##TRAIN


    step_counter=start_iter
    writer = tf.summary.create_file_writer(dir_path)
    
 
    while(step_counter<max_iter):
        


        print("\nStart of iter %d" % (step_counter,))
        print("Learning rate" +str(model.optimizer.lr))


        

        for _, x_batch_train in enumerate(train_dataset):
            
            step_counter+=1
            set_learning_rate(step_counter,model,base_lr,solver_steps,decay_step=decay_step,decay_rate=decay_rate)

            losses = model.train_step(x_batch_train)
            
            kl_loss =losses["kl_loss"]
            loss_reco=losses["loss_reco"]
            total_loss  = losses["total_loss"]
          
     

            


            if step_counter%ckpt_freq ==0:
                model.save_weights(dir_path+"/ckpt"+str(step_counter))
            
            if step_counter%1==0:
               
            
                print("step "+str(step_counter) ) 
                print("kl_loss  : ", kl_loss)
                print("loss_reco  : ", loss_reco)
                print("total_loss  : ", total_loss)
  
              
    

                with writer.as_default():
                    tf.summary.scalar('kl_loss', kl_loss, step=step_counter)
                    tf.summary.scalar('loss_reco',loss_reco , step=step_counter)
                    tf.summary.scalar('total_loss',total_loss , step=step_counter)

          
