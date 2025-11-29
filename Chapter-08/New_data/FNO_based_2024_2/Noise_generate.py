import numpy as np
from PIL import Image
import numpy as np

from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as pyplot

def Random_noise(noise_random_ratio, dim_0, dim_1, dim_2):
    """
    noise_type = "random_noise"  要选择随机噪声中0的概率
    """
    noise = np.random.choice([0, 1], size=[ dim_1, dim_2], p=[noise_random_ratio, 1 - noise_random_ratio])

    all_noise = []
    for i in range(dim_0):
        all_noise.append(noise)

    all_noise = np.array(all_noise)
    return all_noise


def Bulk_noise(bulk_noise_size, num):
    """
    noise_type = "bulk_noise"  bulk噪声的尺寸 可选  bulk_small/bulk_meso/bulk_big
    图像可选：
    bulk_small.jpg/bulk_meso.jpg/bulk_big.jpg
    """
    image = Image.open("../Noise_image/{}.jpg".format(bulk_noise_size))
    image = image.convert('L')
    image.save(".。/Noise_image/{}_gray.jpg".format(bulk_noise_size))
    image = Image.open("../Noise_image/{}_gray.jpg".format(bulk_noise_size)).resize((64, 64))


    data = np.array(image)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] > 200:
                data[i][j] = 1
            else:
                data[i][j] = 0

    all_data = []
    for i in range(num):
        all_data.append(data)
    bulk_mask = np.array(all_data)
    """    
    Square_mask min max: 0 , 1  (0为遮挡区域，1为不遮挡区域）
    Square_mask shape : (84, 64, 64)
    Square_mask type : numpy.ndarray
    注：下方代码 可以展示mask的样貌
    """
    # image = Image.fromarray(data)
    # pyplot.imshow(image)
    # pyplot.show()
    return bulk_mask

