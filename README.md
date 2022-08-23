#  Realtime   Face   Emotion Recognition using  Transfer  Learning  Techniques  And OpenCV 


- Deep Learning architecture will detect a face and then based on facial expression it will decide the emotions of the humans.

- Languages required : Python , OpenCV , Deep Learning - Tensorflow.



**2 Techniques to Find Facial Expressions**

     Camera ,  Bio-Signals (EEG , ECG )



**6 different Emotion Images Data are available :**

     Anger , Disgust , Fear , Happiness , Sadness , Surprise

     Image size  -  416  * 416 gray scale images 

     Total images : 11,473 - Train  size ( 1800 each )

     Total images : 1438 - test size ( 250 each )


**What is Transfer Learning  -** 
      
      Transfer learning is a machine learning method where a model developed for a task is reused 
      as the starting point for a model on a second task.
      
      Imagenet is the dataset which contains 1000 different classes.
      
      Transfer Learning architecture are trained on ImageNet Dataset 
      
      Every architecture will have different training weights for classifying 1000 classes  . 
      
      Every year there is competition of imagenet , where better state of algorithm are developed 
      which classifies the imagenet  dateset with better accuracy. 
      
      All the Transfer learning technique will give o/p for 1000 classes as it is trained 
      on imagenet dataset. 
      

**Types of Transfer learning techniques -**  
	
     AlexNet , VGG16 , VGG19 , ResNet , Inception , Xception , MobileNet , GoogleNet 





# Project Implementation 



**1.Decide the Image pixel size** 

     Resample the images from 416 to 224 image pixels 

     Transfer learning models will accept the images in the form of (224 * 224)pixel images.

     Input_size = (224 , 224)pixels





**2.Output classification problem** 

     Transfer learning technique are trained for classifying 1000 or more different classes.

     Based on your problem statement  decide the no of classes to be classified at the output layer .




**3.Define the ResNet Model and removing the input and o/p layers from ResNet Model and Dont train the weights of layers.**


	resnet = ResNet50(input_shape=image_size+[3],weights='imagenet',include_top=False)
	for layer in inception.layers:
    	layer.trainable=False


	While importing the ResNet50 class, we mention include_top=False. 
	This ensures that we can add our own custom input and output layers according to our data.

	We mention the weights='imagenet'. 
	This means that the Resnet50 model will use the weights it learnt while being trained on the imagenet data.

	Finally, we mention layer.trainable= False in the pretrained model.
	This ensures that the model does not learn the weights again, saving us a lot of time and space complexity.





**4.Adding the input and output layers to ResNet Model and flattening the output layer**


	Converting the multi-dimensional into 1D using Flattening .

	Flattening the input , define the input and output , loss , optimizer , metrics 




**5.Generate  Different Images using Image Generator only for training data not for test data** 


	train_data=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

	The "ImageDataGenerator" will produce 10 images in each iteration of the training.


	training_set=train_data.flow_from_directory(r 'D:\Sandesh\Data Science\Deep Learning\
						Sentiment_Analysis\eINTERFACE_2021_Image\train',
                                                target_size=(224,224),
                                               batch_size=16,
                                               class_mode='categorical'
                                              )

	Takes the path to a directory & generates batches of augmented data.

**6.Fit the model and train by defining epochs, training set , validation set , batch size**


    r = model.fit_generator(generator=training_set,
    steps_per_epoch=len(training_set),
    epochs=50,
    verbose=1,
    validation_data=test_set,
    validation_steps=len(test_set),
    shuffle=True)



**7. PLot the Accuracy and Loss of validation and Training Data**

	# Train and Test / validation Loss
	plt.plot(r.history['loss'],label='train loss')
	plt.plot(r.history['val_loss'],label='validation loss')
	plt.title('Train VS Validation Loss')
	plt.legend()
	plt.show()

	# Train and Test / validation accuracy
	plt.plot(r.history['accuracy'],label='train accuracy')
	plt.plot(r.history['val_accuracy'],label='validation accuracy')
	plt.title('Train VS Validation Accuracy')
	plt.legend()
	plt.show()



**8.OpenCV** 

	cv.cvtColor() -   method is used to convert an image from one color space to another.

	cv.GaussianBlur() - Blur the image

	cvt.Canny() -  Find the Edges from the images.

















