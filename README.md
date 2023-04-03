# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

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

---

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:

```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips

- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
