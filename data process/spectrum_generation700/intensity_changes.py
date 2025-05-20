import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
import math
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TextureGen(object):
    """
        `TextureGen`类用于模拟在随机选择的晶体学方向上具有纹理的XRD谱图。
        纹理表示晶体在不同方向上的取向偏好，例如晶体的择优取向。
        通过应用纹理，模拟得到的XRD谱图的峰值强度将按照一定规律在晶体学方向上进行缩放，
        从而模拟出具有纹理效应的实验XRD谱图。
    """

    def __init__(self, struc, max_texture=0.6, min_angle=10.0, max_angle=80.0):
        """
        Args:
            struc: pymatgen结构对象，用于从中模拟XRD谱图。
            max_texture: 纹理的最大强度。例如，`max_texture=0.6`意味着峰值强度将在其原始强度的+/- 60%范围内进行缩放。
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.max_texture = max_texture
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    #调用calculator对象的get_pattern方法来计算XRD谱图，并返回计算结果
    def pattern(self):
        struc = self.struc
        return self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

    @property
    #从谱图的计算结果中获取X轴（角度）的值，并返回该值
    def angles(self):
        return self.pattern.x

    @property
    #从谱图的计算结果中获取Y轴（强度）的值，并返回该值
    def intensities(self):
        return self.pattern.y

    @property
    #从谱图的计算结果中获取晶面指数的信息，并返回一个列表包含所有晶面指数
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    #这是一个辅助方法，用于将值 v 从区间 [0, 1] 映射到新的区间 [1 - max_texture, 1]。
    #它用于在计算纹理效应时将取值范围映射到指定的区间，以便模拟纹理对XRD谱图峰值强度的影响。
    def map_interval(self, v):
        """
        Maps a value (v) from the interval [0, 1] to
            a new interval [1 - max_texture, 1]
        """

        bound = 1.0 - self.max_texture
        return bound + ( ( (1.0 - bound) / (1.0 - 0.0) ) * (v - 0.0) )

    @property
    #这段代码定义了`textured_intensities`方法，用于模拟纹理对X射线衍射（XRD）谱图峰值强度的影响。
    def textured_intensities(self):
        #从`hkl_list`属性中获取晶面指数列表 `hkls` 和强度值列表 `intensities`
        hkls, intensities = self.hkl_list, self.intensities
        scaled_intensities = []

        # 如果晶格为六方晶系（hexagonal systems），则在其中的四个米勒指数上进行纹理模拟。
        # 确保选取的偏好方向不是零向量，使其不与晶格的对称性冲突。
        if self.struc.lattice.is_hexagonal() == True:
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) # Ensure 0-vector is not used

                
        # 如果晶格为其他类型的晶系，则在其中的三个指数上进行纹理模拟。确保选取的偏好方向不是零向量。
        else:
            check = 0.0
            while check == 0.0:
                preferred_direction = [random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1])]
                check = np.dot(np.array(preferred_direction), np.array(preferred_direction)) # Make sure we don't have 0-vector

        #对于每个晶面指数和对应的峰值强度，计算纹理因子。纹理因子是一个0到1之间的值，用于模拟纹理对峰值强度的增强或减弱。
        #计算纹理因子时，通过计算晶面指数和偏好方向之间的夹角来确定纹理因子的大小。然后，使用`map_interval`方法将纹理因子映射到指定的区间。
        #将纹理因子应用到每个峰值强度上，得到模拟的纹理增强后的强度值，并将其添加到 `scaled_intensities` 列表中。
        for (hkl, peak) in zip(hkls, intensities):
            norm_1 = math.sqrt(np.dot(np.array(hkl), np.array(hkl)))
            norm_2 = math.sqrt(np.dot(np.array(preferred_direction), np.array(preferred_direction)))
            total_norm = norm_1 * norm_2
            texture_factor = abs(np.dot(np.array(hkl), np.array(preferred_direction)) / total_norm)
            texture_factor = self.map_interval(texture_factor)
            scaled_intensities.append(peak*texture_factor)

        return scaled_intensities

    
    """
    这段代码定义了`calc_std_dev`方法，用于根据角度（two theta）和晶体颗粒大小（domain size）计算高斯核函数的标准差。

    该方法使用Scherrer方程来计算峰的全宽半最大值（FWHM），以确定高斯核函数的标准差。
    Scherrer方程与晶体颗粒的大小、晶格常数、Bragg角和X射线波长有关，用于估计X射线衍射峰的晶体颗粒大小。     
    """
    def calc_std_dev(self, two_theta, tau):
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm 晶体颗粒大小，以纳米（nm）为单位。
        Returns:
            standard deviation for gaussian kernel  高斯核函数的标准差的平方
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9 ## 形状因子，与晶体的形状和晶粒大小有关，一般取常数值
        wavelength = self.calculator.wavelength * 0.1 ## 将X射线波长从埃（angstrom）转换为纳米（nm）
        theta = np.radians(two_theta/2.) ## 将角度值从度转换为弧度，并计算Bragg角
        beta = (K * wavelength) / (np.cos(theta) * tau) # 根据Scherrer方程计算峰的FWHM

        ## 将FWHM转换为高斯核函数的标准差
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    @property
    #这段代码定义了`textured_spectrum`方法，用于生成具有纹理效果的模拟X射线衍射（XRD）谱图。                                       
    def textured_spectrum(self):

        angles = self.angles
        #获取经过纹理效果处理后的强度值 `intensities`，即调用了`textured_intensities`方法
        intensities = self.textured_intensities

        #创建一个步长为 `self.max_angle - self.min_angle` 的等间隔步数数组 `steps`，用于映射角度值到最近的数据点
        steps = np.linspace(self.min_angle, self.max_angle, 700)

        #创建一个二维数组 `signals`，其维度为 `(len(angles), steps.shape[0])`，用于保存谱图数据
        signals = np.zeros([len(angles), steps.shape[0]])

        #将处理后的角度值和强度值映射到最近的数据点，并将对应的强度值存储到 `signals` 中
        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang-steps))
            signals[i,idx] = intensities[i]

        # 每一行数据应用高斯核函数（卷积操作），通过`gaussian_filter1d`方法实现
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle)/700
        for i in range(signals.shape[0]):
            row = signals[i,:]
            ang = steps[np.argmax(row)]
            #使用的高斯核函数的标准差 `std_dev` 是根据 `calc_std_dev` 方法计算得到的
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size,
                                             mode='constant')

        # 将所有行的数据（谱图）合并成单个信号 `signal`
        signal = np.sum(signals, axis=0)

        # 归一化信号 using MinMaxScaler
        scaler = MinMaxScaler()
        norm_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten() * 100


        #添加服从正态分布的随机噪声，噪声的标准差为0.25
        noise = np.random.normal(0, 0.25, 700)
        noisy_signal = norm_signal + noise

        # 将信号格式化为适用于卷积神经网络（CNN）的输入格式，即二维数组形式，每个数据点作为一个单独的样本，返回格式化后的信号 `form_signal`
        form_signal = [[val] for val in noisy_signal]

        return form_signal


    """
    这段代码定义了一个名为`main`的函数，用于生成具有纹理效果的模拟X射线衍射（XRD）谱图。

    - `struc`: 输入的pymatgen结构对象，用于模拟XRD谱图。
    - `num_textured`: 要生成的具有纹理效果的XRD谱图的数量。
    - `max_texture`: 最大的纹理强度，用于控制纹理效果的强弱。默认值为0.6，意味着谱图的峰值强度可以增加或减小最多60%。
    - `min_angle`: XRD谱图的最小角度。
    - `max_angle`: XRD谱图的最大角度。
    
    在函数内部，首先创建了一个`TextureGen`对象`texture_generator`，用于生成纹理效果。
    然后，使用列表推导式生成了具有纹理效果的XRD谱图。循环次数由`num_textured`确定。
    
    最后，函数返回具有纹理效果的XRD谱图列表`textured_patterns`。
    """
def main(struc, num_textured, max_texture=0.6, min_angle=10.0, max_angle=80.0):

    texture_generator = TextureGen(struc, max_texture, min_angle, max_angle)

    textured_patterns = [texture_generator.textured_spectrum for i in range(num_textured)]

    return textured_patterns
