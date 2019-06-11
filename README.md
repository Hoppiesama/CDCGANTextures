# CDCGANTextures
Keras / Tensorflow implementation of a Conditional Deep Convolutional Generative Adversarial Network.

This project learns from around 200-300 textures for wood, grass and bricks. With a larger pool I'm sure it would do better.
I do not own any of the textures used in the training set, so do not use them commercially!


***Important!***

This project requires alot of disk space. It outputs lots of images during the training process so you can see the learning process happening. Around 40k epochs was when I found the results were good for me.

Once you are happy with the images generated, grab the appropriate "Generator" file ( .h5 format ) and put it in the active generator folder. This will then allow you to use that generator next time you run the application and simply output 4 images of the type specified.


**Also important!**

You need CUDA, CUDNN, Python3, Tensorflow-gpu, Keras, Matplotlib. I think that's all of them! I dont think tensorflow works with the newer CUDNN and CUDA, so follow the tensorflow gpu stuff on their website : https://www.tensorflow.org/install/gpu.
Pay particular attention to the VERSION NUMBERS on the software part.

Special thanks to Tom Bashford-Rogers for pushing me to do this work during my degree. 
Also more thanks to people who have public repos of similar projects, it was super helpful in learning about this crazy magic deep learning business.
