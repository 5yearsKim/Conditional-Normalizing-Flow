# Conditionial-Normalizing-Flow
Implementing conditional generative model using normalizing flow. 
relative paper:
1. Glow: Generative Flow with Invertible 1x1 Convolutions
https://arxiv.org/abs/1807.03039
2. Guided Image Generation with Conditional Invertible Neural Networks
https://arxiv.org/abs/1907.02392


# Description
Code adapted from https://github.com/chrischute/glow
Adding conditioning layer to affine coupling layer. Tried conditioning for many domain.
Applied style transfer using the property of conditional flow model. (reconstruct image giving different condition in forward and reverse procedure of Glow)


# implementation
utilize celeba dataset
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

1. download celeba dataset and move to /data folder
2. make list of train set and test set
 ```command
  mv data/celeba
  ls *.jpg | tail -n+10 > train_files.txt
  ls *.jpg | head -n10 > test_files.txt
  mv train_files.txt test_files.txt ../..
 ```

3. run code after configuration
```command
python3 train.py
```
4. applying style tranfer with sample data
```command
 python3 inference.py
```

# what you can do?
### super resolution
By training with decimated as a condition, cFlow can successfully generate high resolution images
![Alt text](/figs/SR.jpg)
decimated(input image)/ reconstructed image/ original image
### super resolution with controlled feature
super resolution is ill-posed problem. when resolution is really low, there are many ways to reconstruct the image. we can also control feature of the image by giving additional feature to the model("Smiling" in example below)
![Alt text](/figs/SR_feature.png)
Conditional flow not only reconstructed super blur image to realistic image, but also controlled feature gradiently
### Colorization
implementing colorization by giving gray image as a condition
![Alt text](/figs/Colorization.png)
gray image(input)/ reconstructed image/ original image
### Sketch-to-image
generating image from simple sketch can also be implemented. Conditioin is simply given using canny-edge detection algorithm.(highly sure of better performance if applied with better edge detection model such as HED)
![Alt text](/figs/sketch-to-image.png)
### Style transfer with conditional Flow
filtering image to Normalizing flow with condition image A, and reconstruct image with condition image B, we can somewhat mix two different image together.
![Alt text](/figs/ST_example.png)
here is simple explanation of principle of this style mixing. My FYP paper is of that conditioning to generative model is subtraction of specific information(relatied to condition) from input image
![Alt text](/figs/ST_principle.png)
### image-to-image translation: Modifying feature from given image
by giving feature(such as "smiling", "Pale face" and so on) as a condition and applying same method as Style transfer, I could also modify feature of the image. 
![Alt text](/figs/modifying_feature.png)
