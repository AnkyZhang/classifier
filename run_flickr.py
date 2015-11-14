import classifier
import setup_caffe_network as su
import models as mdl

su.SetupCaffe.gpu_on() #Call this if you have an NVIDIA GPU and it is set up with caffe.

#Set up the caffe model and initialize the classifier.
dir = '../../caffe-master/'
net = mdl.NetModels.setup_flickr_model(dir)
labelsPath = dir + 'examples/finetune_flickr_style/style_names.txt'
cf = classifier.Classifier(net, labelsPath)

#Load an input image and classify it.
imPath = 'input/lunch2.jpeg'
cf.classify_image(imPath)
cf.save_results()

#load as many images as you want.
imPath = 'input/tandem.jpeg'
cf.classify_image(imPath)
cf.save_results()