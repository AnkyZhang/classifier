import caffe


class SetupCaffe:

    def __init__(self, prototxt_path, caffemodel_path, pixel_mean, height, width):
        self.net = caffe.Classifier(prototxt_path, caffemodel_path, mean=pixel_mean,
                               raw_scale=255,
                               channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        self.net.blobs['data'].reshape(1, 3, height, width)

    def get_network(self):
        return self.net

    # GPU mode, call this if you have a GPU set up with caffe.
    @staticmethod
    def gpu_on():
        caffe.set_device(0)
        caffe.set_mode_gpu()
