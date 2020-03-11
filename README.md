# Self-Driving Car - Behavioural Cloning
One of the enthusiastic parts of the Self-driving Car is Behavioural Cloning. Can you imagine how it is fantastic an artificial device to learn driving? It can recognize the surface of the road, and it just knows the steering wheel as the only instrument for avoiding to leave the road. So its effort for being in the way is funny and impressive. 
Before starting this project, you should see the [video demo](https://www.youtube.com/watch?v=yQL1XG5va-8&list=PLChwywmfd8lqhyap8yrjOeALFLkJ5nRTv&index=1) and have the below requirement:

1. Anaconda Python 3.8
2. Unity Car Simulation [Starter Kit Code](https://github.com/udacity/CarND-Term1-Starter-Kit)
3. Unity Car Simulation on [Window x64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

## Files and Usage
| File Name | Comment |
| ------ | ------ |
| `model.py` | Training implementation. |
| `model.json` | Saved the trained model as json file |
| `model.h5` | saved the weight of trained model |
| `drive.py` | Based on this file, you can get a picture from simulation and send suitable command for speed and steering angle of the car. |

So, if you want to train your model on your dataset, you would need to modify the `model.py` file; otherwise, just use `drive.py` for running the trained model on the simulation.
`python drive.py model.json`
Then open the simulator and the wait for loading. Finally, click on the Autonomous mode button and watch delightful and surprising autonomous driving on the road. If you want to train your model keep following steps:
- Generating dataset
- Preprocessing the dataset
- Designing the architecture of the model
- Training and Validating of the model

### Generating dataset
[Unity Car Simulation](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip) has two modes which are indicated by two buttons. One of them is labeled as Training Mode. If you click on it, you will face an environment and a car that you can control by using the Arrow-Key. 

![Welcome page of simulation](https://github.com/PooyaAlamirpour/BehavioralCloning/blob/master/Pictures/welcome-simulation.png)

![Training mode](https://github.com/PooyaAlamirpour/BehavioralCloning/blob/master/Pictures/training-mode.png)

Take a look around the environment and try to keep the car middle of the road. Once you reckon you can drive carefully, press on the record button on top-right, then choose a folder on your system for saving each taken frame, then recording will be started. Try to finish 2 or 3 laps and stop recording. Under the destination folder, lots of images are stored from three installed cameras on the three sides of the car, front, left, and right. I have collected around 31,000 images, and I think it is enough. Now, it is time for designing architecture.

### Designing the architecture of the model
I have designed a model with five convolutional networks. For the first layer, I have gotten average from pixels of each image by dividing per 255. the size of the output of the first layer is 64x64x3. Then a 2D-Conv network is added. I have used `Relu` activation for connecting this 2D-Conv to another layer. Another layer is again 2D-Conv. I have attached five 2D-Conv, which there is `Relu` activation between each one. I have completed my design by adding some flatten and dropout layers to my network.You can see the summary of my design in the below table:

![The summary of model](https://github.com/PooyaAlamirpour/BehavioralCloning/blob/master/Pictures/summary-model.png)

You can see the implemention of this model below:
```python
input_shape = (64,64,3)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer = l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer = l2(0.001)))
model.add(Dense(1, W_regularizer = l2(0.001)))
adam = Adam(lr = 0.0001)
model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
```

The size of the output of this model is one and the type of it is `dense` layer. I have used `Adam` as an optimizer for my model. The below picture demonstrates my model as a graph. For drawing this graph I have used below code:
```python
from keras.utils import plot_model

plot_model(model, to_file=os.path.join('plot', 'model.png'))
```

![Network plot](https://github.com/PooyaAlamirpour/BehavioralCloning/blob/master/Pictures/network-plot.png)

### Preprocessing and Training
Preprocessing the input of the network is one of the crucial techniques that must be considered. Recently I have figured out that there is another essential technique which is called permutation importance. In this technique, you can find which parameter in the dataset is not crucial or has less impact, and one of them has more effective. In this project, we have three cameras, as mentioned. At the beginning of implementing the project, it seems, the center camera which is installed in front of the car is just enough. But after training the model, you will reckon, two other cameras data are essential for keeping the vehicle stay in the middle of the road.So I have used the generated images from all three cameras. Let's look at one of the input images.

![One sample frame](https://github.com/PooyaAlamirpour/BehavioralCloning/blob/master/Pictures/one-sample-image.png)

As you can be noticed, except the surface and side of the road, there are lots of information in the picture such as trees, mountain, lake, etc. We can eliminate all of them to a certain extent by cropping the image. 
```python
cropped = cv2.resize(image[60:140,:], (64,64))
```
I am trying to transfer my experience during the training of this model because I know that you would face up with these issues. 
After training the model, I realized the vehicle tends to drive one side of the road, and it caused the car went out of the way. So I have added the flipping method. By using this method, all the input images are flipped based on the vertical axis. So the input image can be converted to two images, one is original, and another is the flipped. 
```python
new_image = cv2.flip(image,1)
new_angle = angle*(-1)
```
I wanted to add more changes to the input dataset. Because I wanted to make a trained model that can work in an adverse situation. So I have implemented a method that changes the brightness of the input image randomly. In this way, the model is immune to noise.
```python
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
rand = random.uniform(0.3,1.0)
hsv[:,:,2] = rand*hsv[:,:,2]
new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

### Testing the model and troubleshooting
After the model is trained, two types of the file will be generated. The suffix one of them is `h5` and another is `json`. Run the model on the simulation as it was explained. It might be seen the vehicle would start to drive Irregularly on its path. For solving this issue I have added a PID controller for making restrictions for steering angle. This caused the issue is solved to a certain extent.

Thank you for reading my article. Feel free to clone my repository and ask [me](https://www.linkedin.com/in/pooya-alamirpour) any questions. I feel happy If I can help you.


