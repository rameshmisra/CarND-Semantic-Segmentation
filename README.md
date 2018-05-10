# Semantic Segmentation
### Introduction
In this project, pixels of a road in images were labeled road (green) or not-road using a Fully Convolutional Network (FCN) that was based on a modified VGG16 model previously trained and provided.

It is not clear to me if weights and biases of the VGG16 model were frozen (have not figured out how to check); commentary on the discussion board suggested it was not actually frozen. I could have used tf.get_collection() to freeze parameters of the lower layers of the model, but training time was quite reasonable on my system with a GTX 1080 GPU, so I did not pursue that. The size of the trained model did indeed balloon from about 500MB for the VGG16, to about 1.5GB for the complete FCN model.


### FCN model architecture
Identical to the model discussed in the paper on FCNs from UC Berkeley, three 1x1 convolution layers were added to the VGG16 model provided at the end of the third, fourth and final set of convolution layers. Three deconvolution/ upsampling layers followed, and there were two skipped connections added: from the 4th set of convolutional layers to the first deconvolution layer, and from the 3rd set of convolution layers to the 2nd deconvolution layers.

### Parameters for training
After some amount of experimentation, for the training stage I settled on a learning rate of 0.0002, keep probability of 0.75, batch size of 5 and trained the model for 10 epochs. The cross entropy loss declined fairly monotonically from 0.245 after the first epoch of training to 0.043. I could have run for a few more epochs, or further reduce batch size, to realize an even lower loss, but the resultant model appeared to do a reasonable job during inference. I did not yet implement code for validation loss for the training phase (but, given the fairly small training dataset, perhaps the statistic may have limited value).
Note that I used Tensorflow 1.5.

### Results
The model saved using tf.train.Saver() is about 1.5GB large; during running of main.py, choosing to save the model could result in a crash due to memory error; there is an option to choose not save the model after training is completed, which should eliminate this issue. I have submitted all the images from the Inference stage, and they are placed in 3 folders. Processing all the 290 images can take some time, and again an option is offered in the code to not run inference and stop after saving the model.

The model does a reasonable job in accurately labeling the road; in particular see um_000009.png, um_000032.png, um_000077.png, where the model has done a good job of differentiating the two sides of the road from the grassy median, curved road from bus, the road from the shoulder respectively. Cases where the model displays considerable room for improvement include um_000037.png, um_000037.png, among others, where incorrect parts of the image are labeled as road, or the boundaries between road/not-road are erroneous.


### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

