import math
from hopfield_basic import HopfieldBasic
from hopfield import Hopfield
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 50
images = ['image.jpeg', 'images2.jpeg', 'images2.png'] #'images.png'

def process_image(image_name):
    print 'Processing', image_name
    im  = Image.open(image_name)
    bw_im = im.convert('L')
    bw_im = im.resize((IMAGE_SIZE,IMAGE_SIZE))

    o_np_im = np.asarray(bw_im.getdata(band=0))
    vfunc = np.vectorize(lambda x: 0 if x<128 else 1)
    np_im = vfunc(o_np_im)

    v = np_im.ravel()
    return v

def main():

    print 'Starting process'

    # Pre Edit images
    processed_images = map(process_image, images)

    print 'Training Network'
    # Training
    h = Hopfield(IMAGE_SIZE * IMAGE_SIZE)
    h.train(processed_images)

    print 'Testing Network'
    #Test
    output = []
    for x in range(50):
        log = h.test_with_random()

        vfunc2 = np.vectorize(lambda x: 255 if x == 1 else 0)
        log = map(lambda x: vfunc2(x.reshape((IMAGE_SIZE, IMAGE_SIZE))), log)
        output += log

    log = output
    num_x = int(math.sqrt(len(log)+1))
    num_y = int(((len(log)+1) // num_x))

    _, axarr = plt.subplots(num_x+1, num_y+1)

    for p in xrange(len(log)):
        j = p // num_y 
        i = (p - (j * num_y))
        # print i, j, log[p].shape
        axarr[i][j].imshow(log[p])
        axarr[i][j].set_title('{}'.format(p))
    
    plt.show()


if __name__ == '__main__':
    main()
    