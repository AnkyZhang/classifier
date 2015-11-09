import classifier
import setup_caffe_network as su
import numpy as np

def setupModel():

    prototxtPath   = 'models/googlenet/deploy.prototxt'
    caffemodelPath = 'models/googlenet/bvlc_googlenet.caffemodel' #this model comes with caffe
    pixel_mean = np.float32([104.0, 116.0, 122.0]) # ImageNet mean, training set dependent
    height = 224
    width = 224

    caff = su.SetupCaffe(prototxtPath, caffemodelPath, pixel_mean, height, width)
    return caff.get_network()

su.SetupCaffe.gpu_on() #Call this if you have an NVIDIA GPU and it is set up with caffe.

#Set up the caffe model and initialize the classifier.
net = setupModel()
labelsPath = 'models/googlenet/synset_words.txt'
cf = classifier.Classifier(net, labelsPath)

#Load an input image and classify it.
imPath = 'input/lunch2.jpeg'
cf.classify_image(imPath)
cf.save_results()

#load as many images as you want.
imPath = 'input/Oxcart 690.jpg'
cf.classify_image(imPath)
cf.save_results()
