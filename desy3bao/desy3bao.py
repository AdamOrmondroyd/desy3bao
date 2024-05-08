import os
import numpy as np
from cobaya.likelihood import Likelihood

# get path of this file
chi2_dir = os.path.dirname(os.path.realpath(__file__))
chi2_file = "chi2profile_dvdesy3_cosmoplanck18_covcosmolike.csv"

REDSHIFT = 0.835

dist_z = np.linspace(0, 2, 150)


class DESY3BAO(Likelihood):
    def initialize(self):

        # Determine the path to the chi2(alpha) file and load it
        chi2_path = os.path.join(chi2_dir, chi2_file)
        chi2_data = np.loadtxt(chi2_path, delimiter=',', skiprows=1).T

        # Select the part of the data we will actually use.
        # There are various columns denoting different methods for
        # measuring alpha
        chi2_column = 1  # column to get chi2 from
        alpha = chi2_data[0]  # alpha is 0th column
        chi2_alpha = chi2_data[chi2_column]

        # Read the redshift at which to interpolate theory predictions
        redshift = REDSHIFT  # this might need to be changed

        print("Limiting alpha = D_M / r_s values in interpolation:")
        print("alpha min = ", alpha[0])
        print("alpha max = ", alpha[-1])

        # Return data for later
        self.alpha = alpha
        self.chi2_alpha = chi2_alpha
        self.redshift = redshift
        return

    def get_requirements(self):
        return {
            "rdrag": None,  # sound horizon distance
            # TODO: d_m is usually the comoving distance, but the notes say it
            # is the angular diameter distance???
            # going by the notes (and paper [https://arxiv.org/pdf/2107.04646])
            "angular_diameter_distance": {"z": [dist_z]},
        }

    def logp(self, **params_values):

        # Fiducial Planck cosmology rs which was used to compute alpha
        rs_fiducial = 147.6
        # Angular diameter distance at the fiducial cosmology, d_a(zeff=0.835)
        d_a_ficucial = 1616.9
        d_m_fiducial = d_a_ficucial * (1 + self.redshift)

        # load theory distance relations and R_S
        d_m = self.provider.get_angular_diameter_distance(dist_z)  # Mpc
        rs = self.provider.get_param("rdrag")

        # Interpolate the theory at the observed redshift
        d_m_predicted = np.interp(self.redshift, dist_z, d_m)

        # This ratio is what the likelihood is stored as a function of - the
        # ratio of dm/rs to the fiducial value
        alpha_predicted = (d_m_predicted / d_m_fiducial) * (rs_fiducial / rs)

        # Get the chi2 at the measured alpha, clipped to edges of the range
        chi2_alpha_predicted = np.interp(alpha_predicted,
                                         self.alpha,
                                         self.chi2_alpha,
                                         left=self.chi2_alpha[0],
                                         right=self.chi2_alpha[-1])

        # Get the log likelihood from the chi2
        like = -chi2_alpha_predicted / 2.
        return like
