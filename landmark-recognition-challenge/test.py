%%time
fit_generator(self, generator, 
steps_per_epoch=None, 
epochs=1, 
verbose=1, 
callbacks=None, 
validation_data=None, 
validation_steps=None, 
class_weight=None, 
max_queue_size=10, 
workers=1, 
use_multiprocessing=False, 
shuffle=True, 
initial_epoch=0)
hist = model.fit_generator(x_train, y_train,
                 batch_size=8,
                 epochs=10, #Increase this when not on Kaggle kernel
                 verbose=1,  #1 for ETA, 0 for silent
                 callbacks=[annealer],
                 validation_data=(x_val, y_val))
