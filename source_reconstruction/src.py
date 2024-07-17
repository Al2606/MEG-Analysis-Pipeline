# This is a class for source reconstruction analysis.
# The class contains several methods for different steps of the analysis,
# such as performing coregistration between the head shape points and the MRI.
# Overall, this class provides a convenient way to perform source
# reconstruction analysis for multiple subjects using MNE-Python.

"""

!!! To use this code, the data must be stored in the following structure: !!!

- subjects_dir (str): a path to a folder which contains a folders of for all subjects (named: subjectnr), each containing individual subject data
- subjectnr (str): the name of the individual subject data folders which are stored in sucjects_dir (e.g. 'P10')
- bem_model, bem, mri, noisecov_path (str): direct paths to the respective files
- save_path (str): the path to a folder which contains a folders of for all subjects (named: subjectnr), each containing the processed individual subject data
  (can be the same as subjects_dir, if data of one subject should be collected in only one folder)
- The preprocessed .fif MEG datafile must be stored for each subject in the subjects_dir+subjectnr folder as 'MEGdata-raw.fif'. Only the info-file is needed
- The preprocessed and eventually cutted (according to the audio stimulus data) .npy MEG datafile for each subject must be stored in the
  subjects_dir+subjectnr folder as 'meg_data'+'.npy' (e.g. 'meg_audiobook1' ... 'meg_data_part' is a parameter which is specified in the beamformer function)

"""
import os
from pathlib import Path
from typing import Union

import numpy as np
import mne
from mne.coreg import Coregistration
from mne.io import Raw


class SourceRecon:

    def __init__(
        self, subject_id, subjects_dir, bem_model, bem, mri, noisecov_path, save_path, data: Union[Raw, None] = None
    ):
        self.subject_id = subject_id
        self.subjects_dir = Path(subjects_dir)
        self.bem_model = bem_model
        self.bem = bem
        self.mri = mri
        self.noisecov_path = Path(noisecov_path)
        self.save_path = Path(save_path)

        if data is not None:
            self.raw = data
        else:
            self.raw = mne.io.read_raw_fif(
                self.subjects_dir.joinpath(f"{self.subject_id}{os.sep}MEGdata-raw.fif"), preload=True, verbose=True
            )
        self.info = self.raw.info

    def get_ROI(self, mri_path, ROI, verbose=True):
        """
        Given a path to an MRI file and the ROI, return a list of labels for either wholebrain, the cortex, small cortex, and brainstem.

        Args:
            mri_path (str): The path to the MRI file.
            ROI (str): the ROI to get ('wholebrain', 'cortex', 'cortex_small' or 'brainstem').
        Returns:
            labels wholebrain, for the cortex, small cortex, or brainstem.

        """
        if ROI == "wholebrain":
            LABELS = mne.get_volume_labels_from_aseg(mri_path, return_colors=False, atlas_ids=None, verbose=verbose)
            return LABELS
        elif ROI == "cortex":
            cortex = [
                "ctx-rh-middletemporal",
                "ctx-rh-transversetemporal",
                "ctx-rh-superiortemporal",
                "ctx-rh-bankssts",
                "ctx-rh-supramarginal",
                "ctx-rh-insula",
                "ctx-lh-middletemporal",
                "ctx-lh-transversetemporal",
                "ctx-lh-superiortemporal",
                "ctx-lh-bankssts",
                "ctx-lh-supramarginal",
                "ctx-lh-insula",
            ]
            return cortex
        elif ROI == "cortex_small":
            cortex_small = [
                "ctx-rh-transversetemporal",
                "ctx-rh-superiortemporal",
                "ctx-rh-bankssts",
                "ctx-lh-transversetemporal",
                "ctx-lh-superiortemporal",
                "ctx-lh-bankssts",
            ]
            return cortex_small
        elif ROI == "brainstem":
            brainstem = ["Brain-Stem"]
            return brainstem

    def coregistration(self, save=False, show=True, verbose=True):
        """
        Align the MRI with the head shape of the subjects. Automatic alignment, adopted from mne-documentation

        Args:
            save (boolean): if True saves the transfile in the save_path+subjectnr folder. Default: False.
            show (boolean): if True the coregistration figure is shown. Default: True.
        Returns:
            the trans-file for the subjectnr

        """

        # Set up the plot and view parameters for visualizing the alignment
        plot_kwargs = dict(
            subject=self.subject_id,
            subjects_dir=self.subjects_dir,
            surfaces="head-dense",
            dig=True,
            eeg=[],
            meg="sensors",
            show_axes=True,
            coord_frame="meg",
        )
        view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

        # Specify the fiducials to use for coregistration
        fiducials = "estimated"  # get fiducials from fsaverage

        # Create a Coregistration object and fit fiducials
        coreg = Coregistration(self.info, "fsaverage", self.subjects_dir, fiducials=fiducials)
        coreg.fit_fiducials(verbose=verbose)

        # Fit the ICP algorithm with nasion_weight and specified iterations
        coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=verbose)

        # Remove head shape points with a distance greater than 5 mm from the scalp surface
        coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters

        # Refine the ICP registration by increasing the number of iterations and nasion weight
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=verbose)

        # Visualize the alignment
        if show:
            fig = mne.viz.plot_alignment(self.info, trans=coreg.trans, **plot_kwargs)
            mne.viz.set_3d_view(fig, **view_kwargs)

        # Compute the distance between the head shape points and MRI and print the experiments
        dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
        if verbose:
            print(
                f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
                f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
            )

        # Write the computed transformation matrix to a file for later use
        if save:
            mne.write_trans(
                self.save_path.joinpath(f"{self.subject_id}{os.sep}source_level{os.sep}{self.subject_id}-trans.fif"),
                coreg.trans,
                overwrite="True",
            )

        return coreg.trans

    def leadfield(self, ROI, save=False, verbose=True):
        """
        Given a subject number, ROI, the path to the subjects directory, and the path to save files, create a forward solution
        and return it.

        Args:
            ROI (str): The region of interest. Can be 'cortex', 'cortex_small', 'brainstem', or 'wholebrain'.
            save (boolean): if True saves the fwd in the save_path+subjectnr folder. Default: False.

        Returns:
            The forward solution.
        """

        # Get trans_file
        trans_fname = self.coregistration(show=False)

        # Get the labels for the given ROI
        assert ROI == "cortex" or "cortex_small" or "brainstem" or "wholebrain"
        region = self.get_ROI(self.mri, ROI)

        # create volume source space
        src = mne.setup_volume_source_space(
            subject="fsaverage",
            mri=self.mri,
            pos=5.0,
            bem=self.bem_model,
            subjects_dir=self.subjects_dir,
            add_interpolator=True,
            volume_label=region,
            single_volume=False,
            verbose=verbose,
        )

        if verbose:
            print(f"The source space contains {len(src)} spaces and " f"{sum(s['nuse'] for s in src)} vertices")

        # make forward solution
        fwd = mne.make_forward_solution(
            self.info,
            trans=trans_fname,
            src=src,
            bem=self.bem,
            eeg=False,
            meg=True,
            mindist=1.0,
            n_jobs=1,
            verbose=verbose,
        )

        leadfield = fwd["sol"]["data"]
        if verbose:
            print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
            print(
                f"The fwd source space contains {len(fwd['src'])} spaces and "
                f"{sum(s['nuse'] for s in fwd['src'])} vertices"
            )

        if save:
            mne.write_forward_solution(
                self.save_path.joinpath(
                    f"{self.subject_id}{os.sep}source_level{os.sep}{self.subject_id}_{ROI}-fwd.fif"
                ),
                fwd,
                overwrite=True,
            )

        return fwd, src

    def beamformer(self, ROI, meg_data, save=False, verbose=True):
        """
        Performs a beamforming analysis on MEG data of a given subject.

        Args:
            ROI (str): the region for which the beamformer should be computed ('cortex', 'cortex_small', 'brainstem', or 'wholebrain').
            meg_data (Path): the path to the preprocessed, cut, .npy meg data filename
            save (boolean): if True saves the stc data in the save_path+subjectnr folder. Default: False.

        Returns:
            The source estimate array representing the raw beamformed data. Shape: (nsourcepoints x ntimepoints)
        """

        # load .npy meg data of every proband
        raw = self.raw.pick_types(meg=True)
        info = raw.info

        # create RawArray
        megdata = np.load(meg_data)[:248, :]  # 248 magnetometer
        newraw = mne.io.RawArray(megdata, info)  # create raw data

        # create data covariance matrix out of raw data
        data_cov = mne.compute_raw_covariance(newraw, method="empirical", verbose=verbose)

        # read forward solution for the ROI
        assert (
            ROI == "cortex" or "brainstem" or "cortex_small" or "wholebrain"
        )  ##cortex_small and wholebrain not available!

        fwd, src = self.leadfield(ROI)

        # read noise covariance matrix
        noise_cov = mne.read_cov(self.noisecov_path, verbose=verbose)

        # filters for beamformee
        filters = mne.beamformer.make_lcmv(
            info,
            fwd,
            data_cov,
            reg=0.05,
            noise_cov=noise_cov,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank=None,
            verbose=verbose,
        )

        # apply beamformer filter
        stc = mne.beamformer.apply_lcmv_raw(newraw, filters, verbose=verbose)
        stcdata = stc.data

        # save stc
        data_type = os.path.basename(meg_data).split(".")[0]
        if save:
            np.save(self.save_path.joinpath(f"{self.subject_id}{os.sep}source_level{os.sep}{ROI}_{data_type}"), stcdata)
        return stc, src
