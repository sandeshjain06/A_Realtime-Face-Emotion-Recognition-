#  Realtime   Face   Emotion Recognition using  Transfer  Learning  Techniques  And OpenCV 


**6 different Emotion Images Data are available :**

     Anger , Disgust , Fear , Happiness , Sadness , Surprise

     Image size  -  416  * 416 

     Total images : 11,473 - Train  size ( 1800 each )

     Total images : 1438 - test size ( 250 each )


**What is Transfer Learning  -** 
      
      Transfer learning is a machine learning method where a model developed for a task is reused as the starting point 
      for a model on a second task.

**Types of Transfer learning techniques -**  
	
      VGG16 , VGG19 , ResNet , Inception , Xception , MobileNet



**Project Implementation :** 


1.

Input_size = 224 , 224


2.


resnet = ResNet50(input_shape=image_size+[3],weights='imagenet',include_top=False)
for layer in inception.layers:
    layer.trainable=False

While importing the ResNet50 class, we mention include_top=False. 
This ensures that we can add our own custom input and output layers according to our data.

We mention the weights='imagenet'. 
This means that the Resnet50 model will use the weights it learnt while being trained on the imagenet data.

Finally, we mention layer.trainable= False in the pretrained model.
This ensures that the model does not learn the weights again, saving us a lot of time and space complexity.


2.

Flattening the input , define the input and output , loss , optimizer , metrics 

3.

Use ImageDataGenerator - 


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

4. Fit the model and train by defining epochs, training set , validation set , batch size


5.

OPENCV :

cv.cvtColor() -   method is used to convert an image from one color space to another.

cv.GaussianBlur() - Blur the image

cvt.Canny() -  Find the Edges from the images.

















