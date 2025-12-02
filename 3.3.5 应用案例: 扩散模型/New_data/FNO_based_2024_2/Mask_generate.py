import numpy as np
from PIL import Image
import numpy as np

from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as pyplot

def Cloud_mask(num,corrup_rate):
    """
    cloud图像 可以选：
    cloud_right.jpg/cloud_up.jpg
    """
    if corrup_rate==8:
        imagename1 = "531" #8%
        imagename2 = "522" #8% 同的遮盖率
    elif corrup_rate==25:
        imagename1 = "526"  # 25%
        imagename2 = "553"  # 24% 同的遮盖率
    elif corrup_rate==46:
        imagename1 = "732"  # 46%
        imagename2 = "504"  # 44% 同的遮盖率 514 也是46%
    elif corrup_rate == 68:
        imagename1 = "455"  # 68%
        imagename2 = "448"  # 63% 同的遮盖率
    image = Image.open("../Mask_image/"+imagename1+".png")
    image = image.convert('L')
    image.save("../Mask_image/"+imagename1+"_gray.jpg")
    image = Image.open("../Mask_image/"+imagename1+"_gray.jpg").resize((64, 64))

    data = np.array(image)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >200:
                data[i][j] = 0  #200以上是白色，白色是遮挡，黑色是不遮挡
            else:
                data[i][j] = 1

    all_data = []
    for i in range (num):
        all_data.append(data)
    Cloud_mask1 = np.array(all_data)


    image = Image.open("../Mask_image/"+imagename2+".png")
    image = image.convert('L')
    image.save("../Mask_image/"+imagename2+"_gray.jpg")
    image = Image.open("../Mask_image/"+imagename2+"_gray.jpg").resize((64, 64))

    data = np.array(image)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >200:
                data[i][j] = 0  #200以上是白色，白色是遮挡，黑色是不遮挡
            else:
                data[i][j] = 1

    all_data = []
    for i in range (num):
        all_data.append(data)
    Cloud_mask2 = np.array(all_data)
    """    
    Cloud_mask min max: 0 , 1  (0为遮挡区域，1为不遮挡区域）
    Cloud_mask shape : (84, 64, 64)
    Cloud_mask type : numpy.ndarray
    注：下方代码 可以展示mask的样貌
    """
    # image = Image.fromarray(data)
    # pyplot.imshow(image)
    # pyplot.show()

    return Cloud_mask1, Cloud_mask2


def Square_mask(num):
    """
        Square图像 可以选：
        square_right.jpg/square_up.jpg
        """
    image = Image.open("../Mask_image/square_right.jpg")
    image = image.convert('L')
    image.save("../Mask_image/square_right_gray.jpg")
    image = Image.open("../Mask_image/square_right_gray.jpg").resize((64, 64))

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
    Square_mask = np.array(all_data)
    """
    Square_mask min max: 0 , 1  (0为遮挡区域，1为不遮挡区域）
    Square_mask shape : (84, 64, 64)
    Square_mask type : numpy.ndarray
    注：下方代码 可以展示mask的样貌
    """
    # image = Image.fromarray(data)
    # pyplot.imshow(image)
    # pyplot.show()
    return Square_mask


def Strip_mask(num):
    """
        could图像 可以选：
        strip_right.jpg/strip_left.jpg
        """
    image = Image.open("../Mask_image/strip_right.jpg")
    image = image.convert('L')
    image.save("../Mask_image/strip_right_gray.jpg")
    image = Image.open("../Mask_image/strip_right_gray.jpg").resize((64, 64))

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
    Strip_mask = np.array(all_data)
    """
    Strip_mask min max: 0 , 1  (0为遮挡区域，1为不遮挡区域）
    Strip_mask shape : (84, 64, 64)
    Strip_mask type : numpy.ndarray
    注：下方代码 可以展示mask的样貌
    """
    # image = Image.fromarray(data)
    # pyplot.imshow(image)
    # pyplot.show()
    return Strip_mask

