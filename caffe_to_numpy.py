import caffe
import argparse
import pdb
import pickle

def caffe_to_numpy(prototxt_filename, caffemodel_filename):
    net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST) # read the net + weights
    pynet_ = []
    for li in xrange(len(net.layers)):  # for each layer in the net
        print(li)
        layer = {}  # store layer's information
        layer['name'] = net._layer_names[li]
        if ('bn' in layer['name']) or ('norm' in layer['name']):
            print('Not sure about the train or test phase for batch-normalization layer')
            raise NotImplementedError
        # for each input to the layer (aka "bottom") store its name and shape
        #layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in list(net._bottom_ids(li))]
        # for each output of the layer (aka "top") store its name and shape
        #layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) for bi in list(net._top_ids(li))]
        layer['type'] = net.layers[li].type  # type of the layer
        # the internal parameters of the layer. not all layers has weights.
        layer['weight'] = [net.layers[li].blobs[bi].data[...] for bi in xrange(len(net.layers[li].blobs))]
        if not layer['weight']:
            continue
        pynet_.append(layer)
    return pynet_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto-path', help='Model definition (.prototxt) path')
    parser.add_argument('--caffemodel', help='Model data (.caffemodel) path')
    args = parser.parse_args()
    pynet = caffe_to_numpy(args.proto_path, args.caffemodel)
    pickle.dump(pynet, open('saved.p', 'w'))


if __name__ == '__main__':
    main()