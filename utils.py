import numpy as np
import math
import matplotlib.pyplot as plt

def H_base_case(N):
    """
    This finction calculates the transform matrix based of the length N
    of the signal
    N should be power of two
    :param N: len of signal (int)
    :return: DWT transformation matrix
    """
    assert math.log2(N).is_integer()

    I = np.identity(int(N/2))
    upper_v = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    lower_v = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

    upper_k = np.kron(I, upper_v)
    lower_k = np.kron(I, lower_v)

    return np.vstack((upper_k, lower_k))



def kronHaar(H):
    """
    creates a kron prod given a Haar transform matrix
    :param H: Haar transform matix
    :return: Haar transform matrix of the same level but for double the
    size
    """
    assert len(H.shape) == 2
    assert H.shape[0] == H.shape[1]

    I = np.identity(H.shape[0])


    upper_v = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    lower_v = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

    upper_k = np.kron(H, upper_v)
    lower_k = np.kron(I, lower_v)

    return np.vstack((upper_k, lower_k))




def kron_recursive(H, N):
    """
    This function uses an Haar matrix and reescales it until it gets to
    the size NxN
    :param H: Haar matrix
    :param N: final size
    :return: Haar matrix
    """
    if H.shape[0] == N:
        return H

    H = kronHaar(H)
    return kron_recursive(H, N)



def get_H_at_level(level, N):
    """
    This get a Haar matrix N-point and the level gicen by the variable
    level
    :param level: especification of the level
    :param N: size required
    :return: Haar matrix desired
    """
    assert level-1 in [i for i in range(int(math.log2(N)))]
    level = level - 1
    H = H_base_case(int(N/(2**level)))
    H = kron_recursive(H, N)
    return H




def norm(x):
    """
    Computes the norm of a given vector
    :param x: vector or matrix to compute the norm
    :return: norm
    """
    x = x.reshape(-1)
    return np.linalg.norm(x)


def noisyChannel(x, p, l=0, h=255):
    """
    This function adds the efect of an equiprobable noisy channel to
    the vector or matrix x with the probability p. It expects a image to
    be passed
    :param x: vector to distort
    :param p: probability of a value changing
    :param l: low value for a pixel
    :param h: high value for a pixel
    :return: x with noise
    """
    shape = x.shape
    noise = np.random.randint(l, h, shape)
    where = np.where(np.random.binomial(1, p, shape) == 1)
    y = x.copy()
    y[where] = noise[where]
    return y


class HaarFilterBank:
    def __init__(self, image):
        """
        The initial gray scale image has to be passed
        :param image: a square numpy array representing an image
        """
        assert math.log2(image.shape[0]).is_integer()
        assert image.shape[0] == image.shape[1]

        self.N = image.shape[0]
        self.image = image
        self.levels = [i+1 for i in range(int(math.log2(self.N)))]
        self.H = {}
        self.DWT = {}
        self.IDWT = {}
        self.reconstruction_low = {}
        self.reconstruction_high = {}

    def get_Hs(self):
        """
        This function is for generate all the Haar matrices that this image
        can use
        :return: None
        """
        for i in self.levels:
            self.H[i] = get_H_at_level(i, self.N)
        return

    def get_DWT(self):
        """
        This function computes all the DWT and saves them in the dictionary
        if the rtransform matrices are not calculated, it will do it
        :return: None
        """
        if len(self.H) == 0:
            self.get_Hs()
        for i in self.H:
            self.DWT[i] = np.linalg.multi_dot([self.H[i], self.image, self.H[i].T])
        return

    def get_IDWT(self):
        """
        This function computes all the IDWT and saves them in the dictionary
        if the transform matrices are not calculated, it will do it
        :return: None
        """
        if len(self.DWT) == 0:
            self.get_DWT()
        for i in self.H:
            self.IDWT[i] = np.linalg.multi_dot([self.H[i].T, self.DWT[i], self.H[i]])
        return

    def plot_DWT(self, level):
        """
        This plot the DWT
        :param level: which level to plot
        :return: None
        """
        want_save = input('do you want to save this? (y/n): ')
        if want_save == 'y':
            name_of_file = input('enter name of the file without extension: ')
            name_of_file += '.png'
        plt.figure(dpi=300)
        plt.imshow(self.DWT[level], cmap='gray')
        plt.title('DWT of level ' + str(level))
        if want_save == 'y':
            plt.savefig(name_of_file, dpi=300)
        plt.show()
        return

    def plot_IDWT(self, level):
        """
        This plot the IDWT
        :param level: which level to plot
        :return: None
        """
        want_save = input('do you want to save this? (y/n): ')
        if want_save == 'y':
            name_of_file = input('enter name of the file without extension: ')
            name_of_file += '.png'
        plt.figure(dpi=300)
        plt.imshow(self.IDWT[level], cmap='gray')
        plt.title('IDWT of level ' + str(level))
        if want_save == 'y':
            plt.savefig(name_of_file, dpi=300)
        plt.show()
        return

    def createImageFromLow(self, level):
        """
        This function takes the inverse only of the part of the inde
        it saves in a tuple in the self.reconstruction dict
        :param level: level of Wavelet to use
        :return: none
        """
        index = int(self.N/(2**level))
        self.reconstruction_low[level] = np.linalg.multi_dot(
            [self.H[level][:index, :].T,
             self.DWT[level][:index, :index],
             self.H[level][:index, :]]
        )

    def createImageFromHigh(self, level):
        """
        This function takes the inverse only of the part of the inde
        it saves in a tuple in the self.reconstruction dict
        :param level: level of Wavelet to use
        :return: none
        """
        index = int(self.N/(2**level))
        self.reconstruction_high[level] = np.linalg.multi_dot(
            [self.H[level][index:, :].T,
             self.DWT[level][index:, index:],
             self.H[level][index:, :]]
        )

    def plot_reconstruction_high(self, level):
        """
        This plot the reconstruction
        :param tuple_index: tuble index
        :return: None
        """
        want_save = input('do you want to save this? (y/n): ')
        if want_save == 'y':
            name_of_file = input('enter name of the file without extension: ')
            name_of_file += '.png'
        plt.figure(dpi=300)
        plt.imshow(self.reconstruction_high[level], cmap='gray')
        plt.title('reconstruction ' + str(level))
        if want_save == 'y':
            plt.savefig(name_of_file, dpi=300)
        plt.show()
        return

    def plot_reconstruction_low(self, level):
        """
        This plot the reconstruction
        :param tuple_index: tuble index
        :return: None
        """
        want_save = input('do you want to save this? (y/n): ')
        if want_save == 'y':
            name_of_file = input('enter name of the file without extension: ')
            name_of_file += '.png'
        plt.figure(dpi=300)
        plt.imshow(self.reconstruction_low[level], cmap='gray')
        plt.title('reconstruction ' + str(level))
        if want_save == 'y':
            plt.savefig(name_of_file, dpi=300)
        plt.show()
        return

    def makeAllReconstructions(self):
        """
        This function creates all the reconstructions and approximations
        trom truncated IDWT
        :return: None
        """
        for level in self.levels:
            self.createImageFromHigh(level)
            self.createImageFromLow(level)
        return
