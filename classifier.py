import numpy as np
import caffe
import PIL.Image


class Classifier:

    def __init__(self, net, labels_path):
        self.net = net
        self.labels_path = labels_path


    def classify_image(self, imagepath):
        self.image_path = imagepath
        self.net.blobs['data'].data[...] = self.net.transformer.preprocess('data', caffe.io.load_image(self.image_path))
        self.out = self.net.forward()


    def save_results(self):

        probs = self.out['prob'][0]

        # load labels
        labels = np.loadtxt(self.labels_path, str, delimiter='\t')

        # sort top 5 predictions from softmax output
        top_k = self.net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]

        # print the top five to the console
        for i in range(0, 5):
            print 'Class ' + str(top_k[i]) + ', ' + str(probs[top_k[i]]) + ', ' + labels[top_k[i]]
        print ' '

        #Save the classified image named with the highest priority class, probobilty %, and label text.
        im = PIL.Image.open(self.image_path)
        prob = '{: .1f}'.format(probs[top_k[0]] * 100)
        name = labels[top_k[0]]
        outPath = "results/Class " + str(top_k[0]) + ',' + prob + '%, ' + name[10:] + ".jpg"
        im.save(outPath, 'jpeg')

