# -*- coding: utf-8 -*-

# ==============================================================================
#   This source file is part of SBEMimage (github.com/SBEMimage)
#   (c) 2018-2020 Friedrich Miescher Institute for Biomedical Research, Basel,
#   and the SBEMimage developers.
#   (c) Robert A. McLeod (robert.mcleod@hitachi-hightech.com), Hitachi High-tech Canada Inc.
#   This software is licensed under the terms of the MIT License.
#   See LICENSE.txt in the project root folder.
# ==============================================================================

"""
This module provides the commands to operate the SEM. Only the functions
that are actually required in SBEMimage have been implemented.
"""
# Python standard libs
# ---------------------
from typing import Union, Tuple, Sequence
import logging
log = logging.getLogger(__name__)

import os, os.path as path
from configparser import ConfigParser
import concurrent.futures as cf
import time
from time import perf_counter as pc

# External site packages
# ----------------------
import numpy as np
try:
    import hihi
    from hihi.su7000 import (HVState, VacuumMode, AlignMode, ProbeMode, 
        ScanSpeed, ScanState, ScanMethod)
    from hihi.su7000 import (valid_scan_shapes, valid_scan_periods, 
            valid_num_frames, valid_vacuum_targets)
except ImportError:
    # If `hihi` is not present then this module can still be loaded without an 
    # exception but the classes herein cannot be instantiated.
    hihi = None


# Local imports
# -------------
from utils import Error
from sem_control import SEM

class SEM_SU7000(SEM):
    """
    Class for remote SEM control of a Hitachi SU7000 using the HTC `hihi` 
    interface. 
    """

    def __init__(self, config: ConfigParser, sysconfig: ConfigParser):
        SEM.__init__(self, config, sysconfig)

        if hihi is None:
            raise ImportError(f'Cannot instantiate `SEM_SU7000` as `hihi` package not found.')

        try:
            # Create a client for `hihi`
            self._client = hihi.HTCClient()
            # And then get a SU7000 service
            self._su = self._client.attach('SU7000')
        except Exception as e:
            self.error_state = Error.hihi_connect
            self.error_info = str(e)
            return

        # Capture params have to be saved and set when the capture call is made
        self._scan_method: ScanMethod = ScanMethod.CSS  # drift corrected
        self._scan_shape: int = valid_scan_shapes[0]   # 640 x 480 pixels
        self._scan_period: int = valid_scan_periods[0] # 10 s / acquire
        self._num_frames: int = valid_num_frames[0]    # 8 frames / acquire for drift correction

    def load_system_constants(self) -> None:
        """
        Load all SEM-related constants from system configuration.
        """
        SEM.load_system_constants(self)
        # TODO: load any custom Hitachi attributes
        log.info(f'STORE_RES: loaded as {self.STORE_RES}')

    def save_to_cfg(self) -> None:
        """
        Save current values of attributes to config and sysconfig objects.
        """
        SEM.save_to_cfg(self)
        # TODO: save any custom Hitachi attributes

    def turn_eht_on(self) -> bool:
        """
        Turn High Voltage on.
        """
        try:
            self._su.Gun.State = HVState.On
        except Exception as e:
            self.error_state = Error.eht
            self.error_info = str(e)
            return False
        return True

    def turn_eht_off(self) -> None:
        """
        Turn High Voltage off.
        """
        try:
            self._su.Gun.State = HVState.Off
        except Exception as e:
            self.error_state = Error.eht
            self.error_info = str(e)
            return False
        return True

    def is_eht_on(self) -> bool:
        """
        Return `True` if EHT is on.
        """
        return self._su.Gun.State == HVState.On

    def is_eht_off(self) -> bool:
        """
        Return `True` if EHT is off.
        """
        return self._su.Gun.State == HVState.Off

    def get_eht(self) -> float:
        """
        Read current EHT (in kV).
        """
        return self._su.Gun.HighVoltage

    def set_eht(self, target_eht: float) -> None:
        """
        Save the target EHT (in kV, rounded to 1 decimal place) and set
        the actual EHT to this target value.
        """
        target_eht = np.round(target_eht, decimals=1)
        try:
            self._su.Gun.HighVoltage = target_eht
        except Exception as e:
            self.error_state = 306
            self.error_info = str(e)
            return False
        return True

    def has_vp(self) -> bool:
        """
        Return `True` if variable pressure control is supported.
        """
        return True

    def is_hv_on(self) -> bool:
        """
        Return `True` if high voltage is on.
        """
        return self._su.Vacuum.Mode == VacuumMode.High

    def is_vp_on(self) -> bool:
        """
        Return `True` if variable pressure control is set to 'Low'.
        """
        return self._su.Vacuum.Mode == VacuumMode.Variable

    def get_chamber_pressure(self) -> float:
        """
        Read current chamber pressure from SU7000.
        """
        value = self._su.Vacuum.Value # Returns Union[float, str]
        if isinstance(value, float):
            return value
        else:
            return 0.0 # String value of 'High' or similar
       
    def get_vp_target(self) -> int:
        """
        Read current VP target pressure from SU7000.
        """
        log.info('TODO: will have to find closest enumerated value.')
        return self._su.Vacuum.Target

    def set_hv(self) -> bool:
        """Set Vacuum mode to `High`."""
        try:
            self._su.Vacuum.Mode = VacuumMode.High
        except Exception as e:
            self.error_state = Error.hp_hv
            self.error_info = str(e)
            return False
        return True

    def set_vp(self) -> bool:
        """
        Set Variable Pressure to 'Low'.
        """
        try:
            self._su.Vacuum.Mode = VacuumMode.Variable
        except Exception as e:
            self.error_state = Error.hp_hv
            self.error_info = str(e)
            return False
        return True

    def set_vp_target(self, target_pressure: int) -> bool:
        """
        Set the variable pressure target pressure.
        """
        log.info('TODO: will have to find closest enumerated value.')
        target_pressure = np.find_nearest(_valid_vacuum_targets, target_pressure)
        try:
            self._su.Vacuum.Target = target_pressure
        except Exception as e:
            self.error_state = Error.hp_hv
            self.error_info = str(e)
            return False
        return True

    def has_fcc(self) -> bool:
        """Return True if FCC is fitted."""
        return False

    def is_fcc_on(self) -> bool:
        """Return True if FCC is on."""
        return True

    def is_fcc_off(self) -> bool:
        """Return True if FCC is off."""
        return False

    def get_fcc_level(self) -> int:
        """Read current FCC (0-100) from SU7000."""
        return 0

    def turn_fcc_on(self) -> bool:
        """Turn FCC (= Focal Charge Compensator) on."""
        return True

    def turn_fcc_off(self) -> bool:
        """Turn FCC (= Focal Charge Compensator) off."""
        return True

    def set_fcc_level(self, target_fcc_level: int) -> bool:
        """Set the FCC to this target value."""
        return True

    def get_beam_current(self) -> float:
        """
        Read beam current (in pA) from SU7000.
        """
        current = self._su.Gun.EmissionCurrent # μA
        return current * 1e6 # Convert to pA

    def set_beam_current(self, target_current: float) -> bool:
        """
        Save the target beam current (in pA) and set the SEM's beam to this
        target current.
        """
        # Set is not supported for emission current.
        return False

    def get_high_current(self) -> bool:
        """
        Read high current mode from SU7000.
        """
         # This is a QCheckBox so I think it's for "ProbeMode" in our case
        return self._su.ProbeMode == ProbeMode.High

    def set_high_current(self, toggled: bool) -> bool:
        """
        Save the emission current in pA.
        """
        SEM.set_high_current(self, toggled)
        # This is a QCheckBox so I think it's for "ProbeMode" in our case
        try:
            if toggled:
                self._su.ProbeMode = ProbeMode.High
            else:
                self._su.ProbeMode = ProbeMode.Normal
        except Exception as e:
            self.error_state = Error.high_current
            self.error_info = str(e)
            return False
        return True

    def get_aperture_size(self):
        """
        Read aperture size (in μm) from SU7000.
        """
        index = self._su.ObjAp.Index

    def set_aperture_size(self, aperture_size_index: int) -> bool:
        """
        Save the aperture size (in μm) and set the SEM's beam to this
        aperture size.
        """
        # Set is not supported for objective aperture.
        return False


    def apply_frame_settings(self, frame_size_selector: int, pixel_size: float, 
                             dwell_time: float) -> bool:
        """
        Apply the frame settings (frame size, pixel size and dwell time).

        Parameters
        ----------
        frame_size_selector:
            The major (X-axis) of the scan. One of `[640, 1280, 2560, 5120]`.
        pixel_size:
            The pixel size in nm. Field-of-view is then `frame_size_selector * pixel_size`
        dwell_time:
            The pixel dwell time in μs
        """
        mag = self._su.Mag

        
        ret_val1 = self.set_mag(mag)                        # Sets SEM mag
        ret_val2 = self.set_dwell_time(dwell_time)          # Sets SEM scan rate
        ret_val3 = self.set_frame_size(frame_size_selector) # Sets SEM store res

        # Load SmartSEM cycle time for current settings
        scan_speed = self.DWELL_TIME.index(dwell_time)
        # 0.3 s and 0.8 s are safety margins
        self.current_cycle_time = (
            self.CYCLE_TIME[frame_size_selector][scan_speed] + 0.3)
        if self.current_cycle_time < 0.8:
            self.current_cycle_time = 0.8
        return (ret_val1 and ret_val2 and ret_val3)

    def get_frame_size_selector(self):
        """
        Read the current frame size selector from the SEM.
        """
        raise NotImplementedError

    def get_frame_size(self):
        raise NotImplementedError

    def set_frame_size(self, frame_size_selector):
        """
        Set SEM to frame size specified by frame_size_selector.
        """
        raise NotImplementedError

    def get_mag(self) -> int:
        """
        Read current magnification from SEM.
        """
        return self._su.Mag

    def set_mag(self, target_mag: int) -> None:
        """
        Set SEM magnification to target_mag.
        """
        self._su.Mag = target_mag

    def get_pixel_size(self) -> None:
        """
        Read current magnification from the SEM and convert it into
        pixel size in nm.

        Warning
        -------
        Depends on the scan shape being previously set.
        """
        fov = self._su.scan_field_of_view # nm
        return fov[0] / self._scan_shape[0]

        raise NotImplementedError

    def set_pixel_size(self, pixel_size: float) -> None:
        """
        Set SEM to the magnification corresponding to pixel_size. 
        
        Warning
        -------
        Depends on the scan shape being previously set.
        """
        fov = self._su.scan_field_of_view
        mag = self._su.Mag
        current_ps = fov / self._scan_shape
        raise NotImplementedError

    def get_scan_rate(self) -> str:
        """
        Read the current scan rate from the SEM.
        """
        return self._su.Scan.Speed.name

    def set_scan_rate(self, scan_rate_selector: str) -> bool:
        """
        Set SEM to pixel scan rate specified by scan_rate_selector. Should be 
        one of::
            'Rapid1'
            'Rapid2'
            'Fast1'
            'Fast2'
            'Slow1'
            'Slow2'
            'Slow3'
            'Slow4'
            'Slow5'
            'Slow6'
            'Css1'
            'Css2'
            'Css3'
            'Css4'
            'Css5'
            'Css6'
            'Css7'
            'ReduceH'
            'ReduceM'
            'ReduceS'
            'ReduceWideH'
            'ReduceWideS'
            'Custom'
        """
        # TODO: inquire about Custom mode for low-dose usage.

        try:
            self._su.Scan.Speed = scan_rate_selector
        except Exception as e:
            self.error_state = Error.scan_rate
            self.error_info = str(e)
        return True

    def set_dwell_time(self, dwell_time: float):
        """
        Convert dwell time into scan rate and call self.set_scan_rate()
        """
        # Have to set `self._su.Scan.params(method, length, acq_time, n_frames)` lazily.
        self._dwell_time = dwell_time

    def set_scan_rotation(self, angle: float) -> None:
        """
        Set the scan rotation angle (in degrees). Valid range is `(-200°, 200°)`.
        """
        self._su.Scan.Rotation = angle

    def acquire_frame(self, save_path_filename: str, extra_delay: float=0.0) -> bool:
        """
        Acquire a full frame. All imaging parameters must set before calling this
        method.

        Parameters
        ----------
        save_path_filename
            The Hitachi SU7000 files are always saved to a fixed file path, 

                D:\\SemImage\\temp\\C_Image_01.bmp

            so a future is launched that watches the file and makes a copy.

        extra_delay
            Not used.
        """
    
        # Based on usage in `acquisition.py` this method appears to be blocking.
        # Parameters are held lazily until acquire as there are no getters
        self._su._scan.parameters(self._scan_method,
                                  self._scan_shape,
                                  self._scan_period,
                                  self._num_frames)

        micrograph = self._su.Scan.acquire(block=True)
        # SBEMImage doesn't use the returned value.

    def save_frame(self, save_path_filename: str) -> bool:
        """
        Save the frame currently displayed in SU7000.
        """
        # Do we have to implement this? It saves 
        raise NotImplementedError

    def get_wd(self) -> float:
        """
        Return current working distance (in meters).
        """
        value = self._su.WD # mm
        return value * 0.001 # convert from mm to m

    def set_wd(self, target_wd: float) -> bool:
        """
        Set working distance to target working distance (in meters)
        """
        target_wd *= 1000 # convert m to mm
        try:
            self._su.WD = target_wd
        except Exception as e:
            self.error_state = Error.working_distance
            self.error_info = str(e)
            return False
        return True

    def get_stig_xy(self) -> Tuple[float]:
        """
        Read XY stigmation parameters (in %) from SEM, as a tuple.
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return (0.0, 0.0)

    def set_stig_xy(self, target_stig_x: float, target_stig_y: float) -> bool:
        """
        Set X and Y stigmation parameters (in %).
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return True

    def get_stig_x(self) -> float:
        """
        Read X-axis objective stigmation parameter (in %) from SEM.
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return 0.0

    def set_stig_x(self, target_stig_x: float) -> bool:
        """
        Set X-axis objective stigmation parameter (in %).
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return True

    def get_stig_y(self) -> float:
        """
        Read X-axis objective stigmation parameter (in %) from SEM.
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return 0.0

    def set_stig_y(self, target_stig_y: float) -> bool:
        """
        Set Y stigmation parameter (in %).
        """
        # SU7000 does not support objective stigmatism set (yet!)
        return True

    def set_beam_blanking(self, enable_blanking: bool) -> bool:
        """
        Enable beam blanking if enable_blanking == True.
        """
        # SU7000 does not support beam blanking (yet!)
        return True

    def run_autofocus(self) -> bool:
        """
        Run Hitachi autofocus, break if it takes longer than 1 min.
        """
        log.info('TODO: test timeout on autofocus.')
        self._su.Autofocus.start(block=True)
        return True

    def run_autostig(self) -> bool:
        """
        Run Hitachi autostig, break if it takes longer than 1 min.
        """
        log.info('TODO: test timeout on autostigma.')
        self._su.Autostigma.start(block=True)
        return True

    def run_autofocus_stig(self) -> bool:
        """
        Run combined Hitachi autofocus and autostig, break if it takes longer than 1 min.
        """
        self._su.Autofocus.start(block=True)
        self._su.Autostigma.start(block=True)
        return True

    def get_stage_x(self) -> float:
        """
        Read X stage position (in micrometres) from SEM.
        """
        return self._su.Stage.XY[0] * 0.001 # Convert from nm to μm
        

    def get_stage_y(self) -> float:
        """
        Read Y stage position (in micrometres) from SEM.
        """
        return self._su.Stage.XY[1] * 0.001 # Convert from nm to μm

    def get_stage_z(self) -> float:
        """
        Read Z stage position (in micrometres) from SEM.
        """
        return self._su.Stage.Z * 0.001 # convert from nm to to μm

    def get_stage_xy(self) -> Tuple[float]:
        """
        Read XY stage position (in micrometres) from SEM, return as tuple.
        """
        return tuple(self._su.Stage.XY * 0.001)

    def get_stage_xyz(self):
        """
        Read XYZ stage position (in micrometres) from SEM, return as tuple.
        """
        xy = self._su.Stage.XY * 0.001
        return (*xy, self._su.Stage.Z * 0.001)
        
    def move_stage_to_x(self, x: float) -> None:
        """
        Move stage to coordinate x, provided in microns.
        """
        x *= 1e3 # Convert to nm
        # Is supposed to be _synchronous
        self._su.sync('Stage.XY', (x, self._su.Stage.XY[1]))

    def move_stage_to_y(self, y: float) -> None:
        """
        Move stage to coordinate y, provided in microns.
        """
        y *= 1e3 # Convert to nm
        # Is supposed to be _synchronous
        self._su.sync('Stage.XY', (self._su.Stage.XY[0], y))

    def move_stage_to_z(self, z: float) -> None:
        """
        Move stage to coordinate y, provided in microns.
        """
        z *= 1e3 # Convert to nm
        # Is supposed to be _synchronous
        self._su.sync('Stage.Z', z)

    def move_stage_to_xy(self, coordinates: Sequence[float]) -> None:
        """
        Move stage to coordinates x and y, provided as tuple or list
        in microns.
        """
        coordinates = np.array(coordinates) * 1e3
        self._su.sync('Stage.XY', coordinates)

    # def stage_move_duration(self, from_x, from_y, to_x, to_y):
    #     """Calculate the duration of a stage move in seconds using the
    #     motor speeds specified in the configuration."""
    #     duration_x = abs(to_x - from_x) / self.motor_speed_x
    #     duration_y = abs(to_y - from_y) / self.motor_speed_y
    #     return max(duration_x, duration_y) + self.stage_move_wait_interval

    # def reset_stage_move_counters(self):
    #     """
    #     Reset all the counters that keep track of motor moves.
    #     """
    #     self.total_xyz_move_counter = [[0, 0, 0], [0, 0, 0], [0, 0]]
    #     self.failed_xyz_move_counter = [0, 0, 0]
    #     self.slow_xy_move_counter = 0
    #     self.slow_xy_move_warnings.clear()
    #     self.failed_x_move_warnings.clear()
    #     self.failed_y_move_warnings.clear()
    #     self.failed_z_move_warnings.clear()

    # def reset_error_state(self):
    #     """
    #     Reset the error state (to 'no error') and clear self.error_info.
    #     """
    #     self.error_state = Error.none
    #     self.error_info = ''

    def disconnect(self):
        """
        Disconnect from the SEM.
        """
        self._client.detach('SU7000')
