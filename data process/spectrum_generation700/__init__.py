import os
import re

from scipy.signal import find_peaks, filtfilt, resample
import random
import pymatgen as mg
from scipy import signal
from pymatgen.analysis.diffraction import xrd
from skimage import restoration
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate as ip
from pymatgen.core import Structure
from pyts import metrics
import numpy as np
import warnings
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm  # Import tqdm
from pymatgen.io.fleur import FleurInput

from spectrum_generation700 import strain_shifts, uniform_shifts, intensity_changes, peak_broadening, impurity_peaks, mixed
import multiprocessing
from multiprocessing import Pool, Manager
import matplotlib.pyplot as plt
from ase.io import read


class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir, num_spectra=10, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0, 
                 max_strain=0.01, max_shift=0.25, impur_amt=60.0, min_angle=10.0, max_angle=80.0, separate=True):
        """
        Args:
            reference_dir: path to directory containing
                CIFs associated with the reference phases
        """
        self.num_cpu = 50
        self.ref_dir = reference_dir
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.max_shift = max_shift
        self.impur_amt = impur_amt
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate

    def augment(self, phase_info):
        """
        For a given phase, produce a list of augmented XRD spectra.
        By default, 50 spectra are generated per artifact, including
        peak shifts (strain), peak intensity change (texture), and
        peak broadening (small domain size).

        Args:
            phase_info: a list containing the pymatgen structure object
                and filename of that structure respectively.
        Returns:
            patterns: augmented XRD spectra
            filename: filename of the reference phase
        """

        struc, filename = phase_info[0], phase_info[1]
        patterns = []

        if self.separate:
            patterns += strain_shifts.main(struc, self.num_spectra, self.max_strain, self.min_angle, self.max_angle)
            patterns += uniform_shifts.main(struc, self.num_spectra, self.max_shift, self.min_angle, self.max_angle)
            patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
            patterns += impurity_peaks.main(struc, self.num_spectra, self.impur_amt, self.min_angle, self.max_angle)
        else:
            patterns += mixed.main(struc, 5*self.num_spectra, self.max_shift, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.impur_amt, self.min_angle, self.max_angle)

        return (patterns, filename)

    @property
    def augmented_spectra(self):

        phases = []
        for filename in sorted(os.listdir(self.ref_dir)):
            phases.append([Structure.from_file('%s/%s' % (self.ref_dir, filename)), filename])

        with Manager() as manager:

            pool = Pool(self.num_cpu)
            grouped_xrd = pool.map(self.augment, phases)
            sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename
            sorted_spectra = [group[0] for group in sorted_xrd]

            return np.array(sorted_spectra)

    
    def process_cif_file(self, cif_filename):
        cif_name = os.path.splitext(cif_filename)[0]

        try:
            for i in range(len(self.augmented_spectra)):
                for j in range(len(self.augmented_spectra[i])):
                    enhanced_spectrum = self.augmented_spectra[i, j]

                    # Extract "CollCode" number and use it in spectrum_name
                    collcode_match = re.search(r'CollCode(\d+)', cif_name)
                    collcode = collcode_match.group(1) if collcode_match else ""

                    spectrum_name = f"CollCode{collcode}_enhanced{j+1}"
                    spectrum_filename = os.path.join(spectras_folder, f"{spectrum_name}.csv")

                    np.savetxt(spectrum_filename, enhanced_spectrum.squeeze(), delimiter='\t')
                    print(f"Saved spectrum: {spectrum_filename}")

        except Exception as e:
            print(f"Error processing CIF file: {cif_filename} - {e}")
            problematic_files.append(cif_filename)
