import classifier
import setup_caffe_network as su
import models as mdl

su.SetupCaffe.gpu_on() #Call this if you have an NVIDIA GPU and it is set up with caffe.

#Set up the caffe model and initialize the classifier.
dir = '../CommonCaffe/TrainedModels/'
net = mdl.NetModels.setup_flickr_model(dir)
labelsPath = 'models/finetune_flickr_style/style_names.txt'
cf = classifier.Classifier(net, labelsPath)

#Load an input image and classify it.
imPath = 'input/Class 0.jpg'
cf.classify_image(imPath)
cf.save_results()

for i in range(1, 20):
    imPath = 'input/Class ' + str(i) + '.jpg'
    cf.classify_image(imPath)
    cf.save_results()