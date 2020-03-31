# -*- coding: utf-8 -*-

# ==============================================================================
#   SBEMimage, ver. 2.0
#   Acquisition control software for serial block-face electron microscopy
#   (c) 2018-2020 Friedrich Miescher Institute for Biomedical Research, Basel.
#   This software is licensed under the terms of the MIT License.
#   See LICENSE.txt in the project root folder.
# ==============================================================================

"""This module handles all email notifications (status reports and error
messages and remote commands."""

import os
import json
import imaplib
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders, message_from_string

import utils


class Notifications:
    def __init__(self, config, sysconfig,
                 main_controls_trigger, main_controls_queue):
        """Load all settings."""
        self.cfg = config
        self.syscfg = sysconfig
        self.trigger = main_controls_trigger
        self.queue = main_controls_queue

        # E-mail settings from sysconfig (read only)
        self.email_account = self.syscfg['email']['account']
        self.smtp_server = self.syscfg['email']['smtp_server']
        self.imap_server = self.syscfg['email']['imap_server']

        self.user_email_addresses = [self.cfg['monitoring']['user_email'],
                                     self.cfg['monitoring']['cc_user_email']]

        # The password needed to access the e-mail account used by SBEMimage
        # to receive remote commands. The user can set this password in a
        # dialog at runtime
        self.remote_cmd_email_pw = ''

        # Status report
        self.status_report_ov_list = json.loads(
            self.cfg['monitoring']['report_ov_list'])
        self.status_report_tile_list = json.loads(
            self.cfg['monitoring']['report_tile_list'])
        self.send_logfile = (
            self.cfg['monitoring']['send_logfile'].lower() == 'true')
        self.send_additional_logs = (
            self.cfg['monitoring']['send_additional_logs'].lower() == 'true')
        self.send_viewport_screenshot = (
            self.cfg['monitoring']['send_viewport_screenshot'].lower()
            == 'true')
        self.send_ov = (self.cfg['monitoring']['send_ov'].lower() == 'true')
        self.send_tiles = (
            self.cfg['monitoring']['send_tiles'].lower() == 'true')
        self.send_ov_reslices = (
            self.cfg['monitoring']['send_ov_reslices'].lower() == 'true')
        self.send_tile_reslices = (
            self.cfg['monitoring']['send_tile_reslices'].lower() == 'true')
        self.remote_commands_enabled = (
            self.cfg['monitoring']['remote_commands_enabled'].lower() == 'true')

    def save_to_cfg(self):
        self.cfg['monitoring']['user_email'] = self.user_email_addresses[0]
        self.cfg['monitoring']['cc_user_email'] = self.user_email_addresses[1]

        self.cfg['monitoring']['report_ov_list'] = str(
            self.status_report_ov_list)
        self.cfg['monitoring']['report_tile_list'] = json.dumps(
            self.status_report_tile_list)

        self.cfg['monitoring']['send_logfile'] = str(self.send_logfile)
        self.cfg['monitoring']['send_additional_logs'] = str(
            self.send_additional_logs)
        self.cfg['monitoring']['send_viewport_screenshot'] = str(
            self.send_viewport_screenshot)
        self.cfg['monitoring']['send_ov'] = str(self.send_ov)
        self.cfg['monitoring']['send_tiles'] = str(self.send_tiles)
        self.cfg['monitoring']['send_ov_reslices'] = str(self.send_ov_reslices)
        self.cfg['monitoring']['send_tile_reslices'] = str(
            self.send_tile_reslices)
        self.cfg['monitoring']['remote_commands_enabled'] = str(
            self.remote_commands_enabled)

    def send_email(self, subject, main_text, attached_files=[]):
        """Send email to user email addresses specified in configuration, with
        subject and main_text (body) and attached_files as attachments.
        Return (True, None) if email is sent successfully, otherwise return
        (False, error message)."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_account
            msg['To'] = self.user_email_addresses[0]
            if self.user_email_addresses[1]:
                msg['Cc'] =  self.user_email_addresses[1]
            msg['Date'] = formatdate(localtime = True)
            msg['Subject'] = subject
            msg.attach(MIMEText(main_text))
            for f in attached_files:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(open(f, 'rb').read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                                'attachment; filename="{0}"'.format(
                                    os.path.basename(f)))
                msg.attach(part)
            mail_server = smtplib.SMTP(self.smtp_server)
            #mail_server = smtplib.SMTP_SSL(smtp_server)
            mail_server.sendmail(self.email_account,
                                 self.user_email_addresses,
                                 msg.as_string())
            mail_server.quit()
            return True, None
        except Exception as e:
            return False, str(e)

    def send_status_report(self, base_dir, stack_name, slice_counter,
                           main_log, debris_log, error_log, vp_screenshot):
        """Compile a status report and send it via e-mail."""
        attachment_list = []  # files to be attached
        temp_file_list = []   # files to be deleted after email is sent
        missing_list = []     # files to be attached that could not be found

        if self.send_logfile:
            # Generate log file from current content of log in Main Controls
            self.queue.put('GET CURRENT LOG' + main_log)
            self.trigger.s.emit()
            sleep(0.5)  # wait for file to be written
            attachment_list.append(main_log)
            temp_file_list.append(main_log)
        if self.send_additional_logs:
            attachment_list.append(debris_log)
            attachment_list.append(error_log)
        if self.send_viewport_screenshot:
            if os.path.isfile(vp_screenshot):
                attachment_list.append(vp_screenshot)
            else:
                missing_list.append(vp_screenshot)
        if self.send_ov:
            for ov_index in self.status_report_ov_list:
                save_path = os.path.join(base_dir, utils.get_ov_save_path(
                                stack_name, ov_index, slice_counter))
                if os.path.isfile(save_path):
                    attachment_list.append(save_path)
                else:
                    missing_list.append(save_path)
        if self.send_tiles:
            for tile_key in self.status_report_tile_list:
                grid_index, tile_index = tile_key.split('.')
                save_path = os.path.join(base_dir, utils.get_tile_save_path(
                                stack_name, grid_index, tile_index,
                                slice_counter))
                if os.path.isfile(save_path):
                    # If it exists, load image and crop it
                    tile_image = Image.open(save_path)
                    r_width, r_height = tile_image.size
                    cropped_tile_filename = os.path.join(
                        base_dir, 'workspace', 'tile_g'
                        + str(grid_index).zfill(utils.GRID_DIGITS)
                        + 't' + str(tile_index).zfill(utils.TILE_DIGITS)
                        + '_cropped.tif')
                    tile_image.crop((int(r_width/3), int(r_height/3),
                         int(2*r_width/3), int(2*r_height/3))).save(
                         cropped_tile_filename)
                    temp_file_list.append(cropped_tile_filename)
                    attachment_list.append(cropped_tile_filename)
                else:
                    missing_list.append(save_path)
        if self.send_ov_reslices:
            for ov_index in self.status_report_ov_list:
                save_path = os.path.join(base_dir,
                                utils.get_ov_reslice_save_path(ov_index))
                if os.path.isfile(save_path):
                    ov_reslice_img = Image.open(save_path)
                    height = ov_reslice_img.size[1]
                    cropped_ov_reslice_save_path = os.path.join(
                        base_dir, 'workspace', 'reslice_OV'
                        + str(ov_index).zfill(utils.OV_DIGITS) + '.png')
                    if height > 1000:
                        ov_reslice_img.crop(0, height - 1000, 400, height).save(
                            cropped_ov_reslice_save_path)
                    else:
                        ov_reslice_img.save(cropped_ov_reslice_save_path)
                    attachment_list.append(cropped_ov_reslice_save_path)
                    temp_file_list.append(cropped_ov_reslice_save_path)
                else:
                    missing_list.append(save_path)
        if self.send_tile_reslices:
            for tile_key in self.status_report_tile_list:
                grid_index, tile_index = tile_key.split('.')
                save_path = os.path.join(
                                base_dir, utils.get_tile_reslice_save_path(
                                    grid_index, tile_index))
                if os.path.isfile(save_path):
                    reslice_img = Image.open(save_path)
                    height = reslice_img.size[1]
                    cropped_reslice_save_path = os.path.join(
                        base_dir, 'workspace', 'reslice_tile_g'
                        + str(grid_index).zfill(utils.GRID_DIGITS)
                        + 't' + str(tile_index).zfill(utils.TILE_DIGITS)
                        + '.png')
                    if height > 1000:
                        reslice_img.crop(0, height - 1000, 400, height).save(
                            cropped_reslice_save_path)
                    else:
                        reslice_img.save(cropped_reslice_save_path)
                    attachment_list.append(cropped_reslice_save_path)
                    temp_file_list.append(cropped_reslice_save_path)
                else:
                    missing_list.append(save_path)

        # Send report email
        msg_subject = ('Status report for stack ' + stack_name
                       + ': slice ' + str(slice_counter))
        msg_text = 'See attachments.'
        if missing_list:
            msg_text += ('\n\nThe following file(s) could not be attached. '
                         'Please review your e-mail report settings.\n\n')
            for file in missing_list:
                msg_text += (file + '\n')
        success, error_msg = self.send_email(
            msg_subject, msg_text, attachment_list)
        if success:
            self.queue.put('CTRL: Status report e-mail sent.')
            self.trigger.s.emit()
        else:
            self.queue.put('CTRL: ERROR sending status report e-mail: '
                           + error_msg)
            self.trigger.s.emit()

        # Clean up temporary files
        for file in temp_file_list:
            try:
                os.remove(file)
            except Exception as e:
                self.queue.put('CTRL: ERROR while trying to remove '
                               'temporary file: ' + str(e))
                self.trigger.s.emit()

    def send_error_report(self, stack_name, slice_counter, error_state,
                          main_log, vp_screenshot):
        """Send a notification by email that an error has occurred."""
        # Generate log file from current content of log
        self.queue.put('GET CURRENT LOG' + main_log)
        self.trigger.s.emit()
        sleep(0.5)  # wait for file to be written
        if vp_screenshot is not None:
            attachment_list = [main_log, vp_screenshot]
        else:
            attachment_list = [main_log]
        msg_subject = ('Stack ' + stack_name + ': slice '
                       + str(slice_counter) + ', ERROR')
        success, error_msg = self.send_email(
            msg_subject, error_str, attachment_list)
        if success:
            self.queue.put('CTRL: Error notification email sent.')
            self.trigger.s.emit()
        else:
            self.queue.put('CTRL: ERROR sending notification email: '
                           + error_msg)
            self.trigger.s.emit()
        # Remove temporary log file of most recent entries
        try:
            os.remove(main_log)
        except Exception as e:
            self.queue.put('CTRL: ERROR while trying to remove '
                           'temporary file: ' + str(e))
            self.trigger.s.emit()

    def get_remote_command(self):
        """Check email account if command was received from one of the
        allowed email addresses. If yes, return it."""
        try:
            mail_server = imaplib.IMAP4_SSL(self.imap_server)
            mail_server.login(self.email_account, self.remote_cmd_email_pw)
            mail_server.list()
            mail_server.select('inbox')
            result, data = mail_server.search(None, 'ALL')
            if data is not None:
                id_list = data[0].split()
                latest_email_id = id_list[-1]
                # fetch the email body (RFC822) for latest_email_id
                result, data = mail_server.fetch(latest_email_id, '(RFC822)')
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = message_from_string(
                            response_part[1].decode('utf-8'))
                        subject = msg['subject'].strip().lower()
                        sender = msg['from']
                mail_server.logout()
                # Check sender and subject
                # Sender email must be main user email or cc email
                sender_allowed = (self.user_email_addresses[0] in sender
                                  or self.user_email_addresses[1] in sender)
                allowed_commands = ['pause', 'stop', 'continue',
                                    'start', 'restart', 'report']
                if (subject in allowed_commands) and sender_allowed:
                    return subject
                else:
                    return 'NONE'
            else:
                return 'NONE'
        except:
            return 'ERROR'
