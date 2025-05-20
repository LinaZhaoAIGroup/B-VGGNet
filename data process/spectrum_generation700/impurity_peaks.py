import pymatgen as mg
from pymatgen.core import Structure
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from sklearn import preprocessing
import numpy as np
import random
import math
import os
from sklearn.preprocessing import MinMaxScaler



class ImpurGen(object):
    """
    Class used to simulate xrd spectra with broad peaks
        that are associated with small domain size
   
   用于模拟具有宽峰值的 xrd 光谱。与较小的畴尺寸有关
    """

    def __init__(self, struc, impur_amt, ref_dir='References', min_angle=10.0, max_angle=80.0):
        """
        参数:
            min_domain_size：采样的最小畴尺寸（纳米）、 导致最宽的峰值
            max_domain_size：取样的最大畴尺寸（以纳米为单位），产生最窄的峰值
           - `struc`： 用于模拟增强 XRD 的结构。
           - `impur_amt`： 杂质的数量。
           - `ref_dir`： 存储参考光谱的目录（默认为 "References"）。
           - `min_angle`： XRD 图案模拟的最小角度 (2θ)。
           - `max_angle`： XRD 图案模拟的最大角度 (2θ)。
        """
        self.calculator = xrd.XRDCalculator()
        self.struc = struc
        self.impur_amt = impur_amt
        self.ref_dir = ref_dir
        self.min_angle = min_angle
        self.max_angle = max_angle
        # 使用 `XRDCalculator` 类的 `get_pattern` 方法在指定的 2θ 范围内为给定的 `struc`（结构）生成 XRD 。
        #生成的 XRD 图样存储在实例变量 `self.pattern` 中。
        self.pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))

        # 为每个参考相位生成一个单一的干净频谱
        self.saved_patterns = self.clean_specs

    @property
    def clean_specs(self):
        # 通过所有参考结构进行迭代
        ref_patterns = []
        for struc in self.ref_strucs:
    
            # 获取给定结构的XRD谱图在给定的2θ范围内
            pattern = self.calculator.get_pattern(struc, two_theta_range=(self.min_angle, self.max_angle))
            angles = pattern.x
            intensities = pattern.y
    
            # 在给定范围内等分角度，并创建一个与等分后角度相对应的信号矩阵
            steps = np.linspace(self.min_angle, self.max_angle, 700)
            signals = np.zeros([len(angles), steps.shape[0]])
    
            for i, ang in enumerate(angles):
                # 将角度映射到最接近的数据点步骤
                idx = np.argmin(np.abs(ang - steps))
                signals[i, idx] = intensities[i]
    
            # 使用唯一的核对每一行进行卷积
            # 迭代每一行；不能矢量化，因为每一行都要更改核函数
            domain_size = 25.0
            step_size = (self.max_angle - self.min_angle) / 700
            for i in range(signals.shape[0]):
                row = signals[i, :]
                ang = steps[np.argmax(row)]
                std_dev = self.calc_std_dev(ang, domain_size)
                # 高斯核函数期望步长为1 -> 调整 std_dev
                signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size, mode='constant')
    
            # 合并信号
            signal = np.sum(signals, axis=0)
    

            # 使用MinMaxScaler进行归一化
            min_max_scaler = preprocessing.MinMaxScaler()
            norm_signal = min_max_scaler.fit_transform(signal.reshape(-1, 1)).flatten() * 100
            
            

            #归一化后的信号添加到ref_patterns列表中
            ref_patterns.append(norm_signal)
    
        return ref_patterns


    @property
    def impurity_spectrum(self):
        signal = random.choice(self.saved_patterns)#从 self.saved_patterns 列表中随机选择一个信号，并将其赋值给变量 signal
        return signal

    @property
    def ref_strucs(self):
        current_lat = self.struc.lattice.abc#从self.struc 中获取晶格参数，并将其存储在变量 current_lat 中。属性 lattice 是结构对象的一个属性，可以用来获取晶格信息，abc 则是晶格的参数（长度）。
        all_strucs = []#创建了一个空列表 all_strucs，用于存储符合特定条件的结构对象。
        for fname in os.listdir(self.ref_dir):#遍历 self.ref_dir 目录下的文件，迭代处理每个文件名
            fpath = '%s/%s' % (self.ref_dir, fname)#构建了文件的完整路径，将目录名 self.ref_dir 和文件名 fname 连接在一起
            struc = Structure.from_file(fpath)#使用 pymatgen 库中的 Structure.from_file() 方法，从fpath中读取结构数据，并将其存储在变量struc 中
            # 确保没有重复的结构
            if False in np.isclose(struc.lattice.abc, current_lat, atol=0.01):#确保读取的结构与当前结构的晶格参数不完全相同。np.isclose() 函数用于比较两个数组中的元素是否接近，atol=0.01 设置了一个绝对误差的阈值，如果晶格参数之间的差异小于该阈值，则认为它们相等
                all_strucs.append(struc)#将符合条件的结构对象 struc 添加到 all_strucs 列表中
        return all_strucs

    @property
    #返回 XRD 图谱的角度数据。在这个类中，XRD 图谱的角度数据存储在 self.pattern.x 中。由于 pattern 是一个 XRD 图谱对象，使用 .x 可以获取角度数据
    def angles(self):
        return self.pattern.x

    @property
    #返回 XRD 图谱的强度。在这个类中，XRD 图谱的强度数据存储在 self.pattern.y 中。由于 pattern 是一个 XRD 图谱对象，使用 .y 可以获取强度数据。
    def intensities(self):
        return self.pattern.y

    @property
    #返回 XRD 图谱中的晶面指数列表（hkl）。在这个类中，XRD 图谱的晶面指数数据存储在 self.pattern.hkls 中。该属性返回一个列表，其中包含每个峰的晶面指数，每个晶面指数表示为一个字典，具有键 'hkl'。由于 pattern 是一个 XRD 图谱对象，使用 .hkls 可以获取晶面指数数据。
    def hkl_list(self):
        return [v[0]['hkl'] for v in self.pattern.hkls]

    def calc_std_dev(self, two_theta, tau):
        """
        基于two_theta和domain size计算高斯核函数的标准差（standard deviation）。
        Args:
            two_theta: 是在 2θ 空间的角度值（单位为度）
            tau: 领域尺寸（domain size），单位为纳米（nm）
        Returns:
            高斯核函数的标准差
        """
        ## Calculate FWHM based on the Scherrer equation
        """
        Scherrer 方程计算 FWHM 是通过以下步骤进行的：
          计算形状因子 K，其值为 0.9（常用的形状因子）。
          将波长 wavelength 转换为纳米单位（Angstrom 转换为 nm）。
          计算 Bragg 角度 theta，其为输入角度 two_theta 的一半（以弧度为单位）。
          计算 beta，其为 K 乘以波长并除以 cos(theta) 与领域尺寸 tau 的乘积，单位为弧度。
          然后，将 FWHM 转换为高斯核函数的标准差 sigma。标准差 sigma 是计算高斯核函数的一个重要参数，用于描述高斯分布的宽度。
          这里的计算是通过简单的数学转换得到的。
          
          最后，方法返回 sigma 的平方作为高斯核函数的标准差（因为在高斯核函数中需要使用标准差的平方值）。
        """
        K = 0.9 ## shape factor
        wavelength = self.calculator.wavelength * 0.1 ## angstrom to nm
        theta = np.radians(two_theta/2.) ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2


    @property
    def spectrum(self):
        """
        这段代码定义了一个名为 `spectrum` 的方法，用于生成模拟的X射线衍射（XRD）谱图。
        """
        #获取原始的角度值 `angles` 和强度值 `intensities`。
        angles = self.angles
        intensities = self.intensities
        
        #创建一个步长为 `self.max_angle - self.min_angle` 的等间隔步数数组 `steps`，用于后续映射角度值到最近的数据点。
        steps = np.linspace(self.min_angle, self.max_angle, 700)

        #创建一个二维数组 `signals`，其维度为 `(len(angles), steps.shape[0])`，用于保存谱图数据。
        signals = np.zeros([len(angles), steps.shape[0]])

        #将原始数据中的每个角度值映射到最近的数据点，并将对应的强度值存储到 `signals` 中。
        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang-steps))
            signals[i,idx] = intensities[i]

        # 对每一行数据应用高斯核函数（卷积操作），
        # Iterate over rows; not vectorizable, changing kernel for every row
        #通过 `gaussian_filter1d` 方法实现。使用的高斯核函数的标准差 `std_dev` 是根据 `calc_std_dev` 方法计算得到的。
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle)/700
        for i in range(signals.shape[0]):
            row = signals[i,:]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size,
                                             mode='constant')

        # 将所有行的数据（谱图）合并成单个信号 `signal`
        signal = np.sum(signals, axis=0)

        # Normalize signal
        signal = 100 * signal / max(signal)

        # 添加杂质信号，通过 impurity_spectrum 获取一个随机的杂质信号，并将其与原信号相加。杂质信号的强度由随机生成的 impurity_magnitude 决定。
        impurity_signal = self.impurity_spectrum
        impurity_magnitude = random.choice(np.linspace(0, self.impur_amt, 100))
        impurity_signal = impurity_magnitude * impurity_signal / max(impurity_signal)
    
        # Define the scaler object
        scaler = MinMaxScaler()
    
        # Use the scaler to normalize impurity_signal
        impurity_signal = scaler.fit_transform(impurity_signal.reshape(-1, 1)).flatten() * 100
        signal += impurity_signal
    
        # 添加噪声，通过在信号上添加服从正态分布的随机噪声，噪声的标准差为 0.25。
        # 归一化噪声信号 using MinMaxScaler
        noise = np.random.normal(0, 0.25, 700)
        noise = scaler.fit_transform(noise.reshape(-1, 1)).flatten()
    
        noisy_signal = signal + noise
    
        # 将信号格式化为适用于卷积神经网络（CNN）的输入格式，即二维数组形式，每个数据点作为一个单独的样本，返回格式化后的信号 `form_signal`.
        form_signal = [[val] for val in noisy_signal]
    
        return form_signal


def main(struc, num_impure, impur_amt=70.0, min_angle=10.0, max_angle=80.0, ref_dir='References'):

    impurity_generator = ImpurGen(struc, impur_amt, ref_dir, min_angle, max_angle)

    impure_patterns = [impurity_generator.spectrum for i in range(num_impure)]

    return impure_patterns

"""
这段代码定义了一个名为 `main` 的函数，用于生成模拟的掺杂质XRD谱图。
首先，函数接受一些输入参数：
- `struc`: 一个晶体结构，用于生成模拟谱图。
- `num_impure`: 表示希望生成多少个掺杂谱图。
- `impur_amt`: 控制掺杂谱图中杂质信号的强度，默认为70.0。
- `min_angle`: X射线衍射的最小角度，默认为10.0。
- `max_angle`: X射线衍射的最大角度，默认为80.0。
- `ref_dir`: 存放参考结构文件的目录，默认为'References'。

然后，函数实例化一个名为 `impurity_generator` 的 `ImpurGen` 类对象，并将输入参数传递给它。`ImpurGen` 类用于生成模拟的X射线衍射谱图，其中包含了一些模拟和处理谱图的方法。
接下来，函数通过循环调用 `impurity_generator` 的 `spectrum` 方法来生成多个掺杂谱图，并将这些谱图存储在 `impure_patterns` 列表中。
最后，函数返回 `impure_patterns` 列表，包含了生成的多个掺杂谱图。
总体而言，这个函数用于生成一组掺杂X射线衍射谱图，可以用于模拟不同杂质含量的实验数据，进而进行后续数据分析和研究。
"""