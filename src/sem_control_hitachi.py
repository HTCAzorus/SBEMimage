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
import imageio
import numpy as np
try:
    import hihi
    from hihi.su7000 import (HVState, VacuumMode, AlignMode, ProbeMode, 
        ScanSpeed, ScanState, ScanMethod)
    from hihi.su7000.device import (valid_scan_shapes, valid_scan_periods, 
            valid_num_frames, valid_vacuum_targets, find_nearest)
    from framestream import color_print as cp
except ImportError:
    # If `hihi` is not present then this module can still be loaded without an 
    # exception but the classes herein cannot be instantiated.
    hihi = None


# Local imports
# -------------
from utils import Error
from sem_control import SEM

_DWELL_MAP = { # (shape, dwell): period
    (640,   25):  10, 
    (640,   50):  20,
    (640,  100):  40,
    (1280,  25):  20,
    (1280,  50):  40,
    (1280, 100):  80,
    (2560,  25):  40,
    (2560,  50):  80,
    (2560, 100): 160,
    (5120,  25):  80,
    (5120,  50): 160,
    (5120, 100): 320
}

class SEM_SU7000(SEM):
    """
    Class for remote SEM control of a Hitachi SU7000 using the HTC `hihi` 
    interface. 
    """


    def __init__(self, config: ConfigParser, sysconfig: ConfigParser):
        SEM.__init__(self, config, sysconfig)

        # WARNING: calls to `log` do not work until `hihi.HTCClient` is constructed
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
        
        # log.info(cp.y('Loading syscfg'))
        # log.info(cp.y('=============='))
        # log.info(cp.y({section: dict(sysconfig[section]) for section in sysconfig.sections()}))
        # log.info(cp.lr('\n\nLoading config'))
        # log.info(cp.lr('============='))
        # log.info(cp.lr({section: dict(config[section]) for section in config.sections()}))

        # Try to force the scan to be running as when frozen it breaks most of 
        # the extcon functionality.
        self._su.sync('Scan.State', ScanState.Run)
        self._su._debug_mode = True
        # Capture params have to be saved and set when the capture call is made
        self._scan_method: ScanMethod = ScanMethod.Slow
        self._scan_shape: int = valid_scan_shapes[0]   # 640 x 480 pixels
        # self._scan_period: int = valid_scan_periods[0] # 10 s / acquire
        self._dwell_time: float = self.DWELL_TIME[0]
        self._num_frames: int = valid_num_frames[0]    # 8 frames / acquire for drift correction

    def load_system_constants(self) -> None:
        """
        Load all SEM-related constants from system configuration.
        """
        SEM.load_system_constants(self)
        # TODO: load any custom Hitachi attributes
        # log.info(f'STORE_RES: loaded as {self.STORE_RES}')
        # log.info(f'')
        # log.info(f'DWELL_TIME: loaded as {self.DWELL_TIME}')

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
            # self._su.Gun.State = HVState.On
            self._su.sync('Gun.State', HVState.On)
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
            self._su.sync('Gun.State', HVState.Off)
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
            self._su.sync('Gun.HighVoltage', target_eht)
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
            self._su.sync('Vacuum.Mode', VacuumMode.High)
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
            # self._su.Vacuum.Mode = VacuumMode.Variable
            self._su.sync('Vacuum.Mode', VacuumMode.Variable)
        except Exception as e:
            self.error_state = Error.hp_hv
            self.error_info = str(e)
            return False
        return True

    def set_vp_target(self, target_pressure: int) -> bool:
        """
        Set the variable pressure target pressure.
        """
        # log.info('TODO: will have to find closest enumerated value.')
        target_pressure = find_nearest(target_pressure, valid_vacuum_targets)
        try:
            # self._su.Vacuum.Target = target_pressure
            self._su.sync('Vacuum.Target', target_pressure)
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
                self._su.sync('ProbeMode', ProbeMode.High)
                # self._su.ProbeMode = ProbeMode.High
            else:
                self._su.sync('ProbeMode', ProbeMode.Normal)
                # self._su.ProbeMode = ProbeMode.Normal
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
        return self.APERTURE_SIZE[index]

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
        ret_val1 = self.set_pixel_size(pixel_size)
        ret_val2 = self.set_dwell_time(dwell_time)          # Sets SEM scan rate
        ret_val3 = self.set_frame_size(frame_size_selector) # Sets SEM store res

        # Load cycle time/scan_period for current settings
        #     TODO: allow faster scans via Fast? 
        # self._scan_method = ScanMethod.Slow
        scan_speed = self.DWELL_TIME.index(dwell_time)
        self.current_cycle_time = self.CYCLE_TIME[frame_size_selector][scan_speed]

        return (ret_val1 and ret_val2 and ret_val3)

    def get_frame_size_selector(self) -> int:
        """
        Returns the index of the current frame shape, as stored in self.STORE_RES
        """
        major_length = [shape[0] for shape in self.STORE_RES]
        return major_length.index(self._scan_shape)

    # NOT USED:
    # def get_frame_size(self):
    #     return self._scan_shape

    def set_frame_size(self, frame_size_selector: int) -> bool:
        """Set SEM to frame size specified by frame_size_selector."""
        try:
            self._scan_shape = valid_scan_shapes[frame_size_selector]
        except IndexError:
            self.error_state = Error.frame_size
            self.error_info = f'{frame_size_selector} is not a valid shape index.'
            return False
        return True

    def get_mag(self) -> int:
        """
        Read current magnification from SEM.
        """
        return self._su.Mag

    def set_mag(self, target_mag: int) -> None:
        """
        Set SEM magnification to target_mag.
        """
        self._su.sync('Mag', target_mag)

    def get_pixel_size(self) -> None:
        """
        Read current magnification from the SEM and convert it into
        pixel size in nm.

        Warning
        -------
        Depends on the scan shape being previously set.
        """
        fov = self._su.scan_field_of_view # nm
        # self._scan_shape is the major (X-axis) only
        return fov[1] / self._scan_shape

    def set_pixel_size(self, pixel_size: float) -> None:
        """
        Set SEM to the magnification corresponding to pixel_size in nanometers.
        
        Warning
        -------
        Depends on the scan shape being previously set.
        """
        self._su.scan_field_of_view = pixel_size * self._scan_shape

    def get_scan_rate(self) -> int:
        """
        Read the index from the list of valid dwell times.
        """
        return self._scan_method.value


    def set_scan_rate(self, scan_rate_selector: int) -> bool:
        """
        Set SEM to pixel scan rate specified by scan_rate_selector integer index.
        """
        # TODO: inquire with HHT about Custom mode for low-dose usage.
        self._scan_method = ScanMethod(scan_rate_selector)
        return True

    def get_dwell_time(self) -> float:
        """Return dwell time in microseconds."""
        return self._dwell_time

    def set_dwell_time(self, dwell_time: float) -> bool:
        """
        Convert dwell time into scan rate and call self.set_scan_rate()
        """
        log.info(cp.m(f'Setting dwell time to {dwell_time} mapping into list {self.DWELL_TIME}'))
        # TODO: should set scan_period via some lookup table?
        # Have to set `self._su.Scan.params(method, length, acq_time, n_frames)` lazily.
        dwell_time = find_nearest(dwell_time, self.DWELL_TIME)
        if not (self._scan_shape, self._dwell_time) in _DWELL_MAP:
            self.error_state = Error.dwell_time
            self.error_info = f'{dwell_time} is not a valid dwell time for scan shape {self._scan_shape}'
            return False
        self._dwell_time = dwell_time
        return True

    def set_scan_rotation(self, angle: float) -> None:
        """
        Set the scan rotation angle (in degrees). Valid range is `(-200°, 200°)`.
        """
        self._su.sync('Scan.Rotation', angle)

    def acquire_frame(self, save_path_filename: str, extra_delay: float=0.0) -> bool:
        """
        Acquire a full frame. All imaging parameters must set before calling this
        method.

        Parameters
        ----------
        save_path_filename
            The absolute path to which the file will be saved. Note that SU7000
            always saves to '.bmp' so extensions such as '.tif' will be removed.
        extra_delay
            Not used.
        """
        # Based on usage in `acquisition.py` this method appears to be blocking.
        # Parameters are held lazily until acquire as there are no getters 
        # in SU7000 external control API.
        scan_period = _DWELL_MAP[self._scan_shape, self._dwell_time]
        try:
            self._su.sync('Scan.params', self._scan_method,
                                self._scan_shape,
                                scan_period,
                                self._num_frames)
        except (ValueError, SyntaxError) as e:
            self.error_state = Error.grab_image
            self.error_info = f'Failed to set scan parameters to {self._scan_method, self._scan_shape, self._scan_period, self._num_frames} due to {e}'
            return False
        try:
            micrograph = self._su.sync('Scan.acquire', save_path_filename)
        except SyntaxError as e:
            self.error_state = Error.grab_image
            self.error_info = f'Failed to capture scan to path: {save_path_filename}'
            return False
        # self._su.Scan.State
        # SBEMImage doesn't use the returned value.
        # However, SBEMImage expects all filenames to be .TIF extension, so 
        # we load up the BMP and save as a TIFF.
        # (We could save micrograph directly too if it's not float32)
        bmp_filename = path.splitext(save_path_filename)[0] + '.bmp'
        imageio.imwrite(save_path_filename, imageio.imread(bmp_filename))
        return True

    def save_frame(self, save_path_filename: str) -> bool:
        """
        Save the frame currently displayed in SU7000.
        """
        self.error_state = Error.grab_image
        self.error_info = 'Unsupported: Hitachi SU7000 does not support saving current frame in GUI.'
        return False

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
            self._su.sync('WD', target_wd)
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
        self._sy.sync('Autofocus.start')
        return True

    def run_autostig(self) -> bool:
        """
        Run Hitachi autostig, break if it takes longer than 1 min.
        """
        self._sy.sync('Autostigma.start')
        return True

    def run_autofocus_stig(self) -> bool:
        """
        Run combined Hitachi autofocus and autostig, break if it takes longer than 1 min.
        """
        self._sy.sync('Autofocus.start')
        self._sy.sync('Autostigma.start')
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
        xy = self._su.Stage.XY
        # print(cp.g(f'Got XY = {xy} nm'))
        xy *= 0.001  # convert from nm to to μm
        return tuple(xy)

    def get_stage_xyz(self):
        """
        Read XYZ stage position (in micrometres) from SEM, return as tuple.
        """
        xyz = self._su.Stage.XYZTR[0:3]
        # print(cp.g(f'Got XYZ = {xyz} nm'))
        xyz *=  0.001 # convert from nm to to μm
        return tuple(xyz)

        
    def move_stage_to_x(self, x: float) -> None:
        """
        Move stage to coordinate x, provided in microns.
        """
        x *= 1e3 # Convert to nm
        # print(cp.b(f'Move stage-X to {x} nm'))
        # Is supposed to be asynchronous?
        self._su.sync('Stage.XY', (x, self._su.Stage.XY[1]))

    def move_stage_to_y(self, y: float) -> None:
        """
        Move stage to coordinate y, provided in microns.
        """
        y *= 1e3 # Convert to nm
        # Is supposed to be asynchronous?
        # print(cp.b(f'Move stage-X to {y} nm'))
        self._su.sync('Stage.XY', (self._su.Stage.XY[0], y))

    def move_stage_to_z(self, z: float) -> None:
        """
        Move stage to coordinate y, provided in microns.
        """
        z *= 1e3 # Convert to nm
        # Is supposed to be asynchronous?
        self._su.sync('Stage.Z', z)

    def move_stage_to_xy(self, coordinates: Sequence[float]) -> None:
        """
        Move stage to coordinates x and y, provided as tuple or list
        in microns.
        """
        coordinates = np.array(coordinates) * 1e3
        # print(cp.b(f'Move stage to {coordinates} nm'))
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
