# **Behavioral Cloning Project**

The goals of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road

[//]: # "Image References"
[netron]: ./writeup-img/model-visualize.png "Model Visualize"
[center-camera-1]: ./writeup-img/center-camera-1.jpg "Center camera 1"
[left-camera-1]: ./writeup-img/left-camera-1.jpg "Left camera 1"
[right-camera-1]: ./writeup-img/right-camera-1.jpg "Right camera 1"

### Submitted Files

My project includes the following files:

- [model.py](model.py) containing the script to create and train the model
- [drive.py](drive.py) for driving the car in autonomous mode
- [model.h5](model.h5) containing a trained convolution neural network
- [writeup_report.md](writeup_report.md) which provides a summary of the project.

Using the Udacity provided simulator and [drive.py](drive.py) file, the car can be driven autonomously around the track by executing:

```sh
python drive.py model.h5
```

The [model.py](model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture

For my model I used architecture similar to nVidia self-driving car architecture which was presented in Udacity materials. I was trying to use LeNet architecture first, but it does not bring the required results.

Model defined on lines 20-34. I used [netron.app](https://netron.app/) to visualize it:

![alt text][netron]

The model includes several convolutional layers with ReLU activations to introduce nonlinearity. The data is normalized in the model using a Keras Lambda layer. Additionally, to improve model generalization, a cropping layer is included which removed unnecessary information from input images.

For tuning the model, I used an Adam Optimizer and Mean Squared Error for calculating loss(line 101).

### Training Strategy

Training data was chosen to keep the vehicle driving on the road. I used a combination of several center lane driving and recovering from the left and right sides of the road.
To obtain more accurate data, I used a mouse to control the steering of the car, which provided more fluid and precise steering angle data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set(line 95). Final split was 80%/20%.

There were a few spots where the vehicle fell off the track, so I added several recovery records for these specific spots.

Also, to improve model training and generalization, I am also added information from left and right cameras to my dataset with small correction to steering angle(in my case I decided to adjust angle by 0.2)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Creation of the Training Set & Training Process

Here is an example of centered camera image which was used in training:
![alt text][center-camera-1]

For this particular image steering angle was -0.2048611.

I recorded full lap several times and then picked the best attempt to include into my dataset.

Also for the example of center image above I want to show how left and right cameras looks:

Left camera:

![alt text][left-camera-1]

Right camera:

![alt text][right-camera-1]

In final dataset, on which my model was trained, the steering angle was adjusted and added as follows, respectively: -0.0048611 and -0.4048611

While attempting to augment the data by flipping the images, I noticed that it added a strange behavior to my model. Specifically, it caused the simulator to randomly steer to the left in certain areas. As a result, I decided not to include image flipping in my data augmentation strategy.

### Final result

You can see how my model performed on track one in this [video](video.mp4)
