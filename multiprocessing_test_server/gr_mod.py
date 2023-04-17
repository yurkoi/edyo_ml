#!/usr/bin/env python3

import nest_asyncio
import json
import os
import asyncio
import numpy as np
import concurrent.futures
import logging
from multiprocessing import Process
from threading import Thread


import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib, GstApp

nest_asyncio.apply()

connections = 0


class VideoGStream(object):
    def __init__(self, sdp=None):
        global connections
        if sdp:
            self.sdp = sdp
        else:
            logging.error("SDP info not received")
            
        self.is_running = False
        self.idx = 0
        Gst.init(None)
        connections += 1
        self.connection = connections
        command = " filesrc location='{}' ! sdpdemux ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videorate ! video/x-raw,format=(string)BGR,framerate=15/1 ! videoconvert ! appsink max-buffers=1 drop=true sync=false  name=appsink{}".format(self.sdp, self.connection)
        self.pipeline = Gst.parse_launch(command)
        self.cur_frame = None  # last returned frame

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.pipeline.set_state(Gst.State.NULL)
            self.main_loop.quit()
            self.cur_frame = None

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.idx = 0

            self.main_loop = GLib.MainLoop()  # cause error if create in __init__
            main_thread = Thread(target=self.main_loop.run)
            main_thread.daemon = True
            main_thread.start()
            self.pipeline.set_state(Gst.State.PLAYING)
            self.appsink = self.pipeline.get_by_name("appsink{}".format(self.connection))  # cause error if create in __init__

    def get_sample(self):
        sample = None
        if self.is_running:
            sample = self.appsink.try_pull_sample(Gst.SECOND)
            if sample is not None:
                self.idx += 1
        return sample

    @staticmethod
    def sample_to_array(sample, rgb=False):
        if sample is None:
            return None  # np.zeros((100, 100, 3), dtype=np.uint8)
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        # create image (nd.array) from gst stream
        array = np.ndarray(
            (height, width, 3),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        # invert bgr to rgb
        if rgb:
            array = array[:, :, ::-1]
        return array

    # loop for reading frames
    def read_and_save(self):

        while self.is_running:
            sample = self.get_sample()
            frame = self.sample_to_array(sample)
            if frame is not None:
                self.cur_frame = frame.copy()


class GrRecognizer(Process):

    def __init__(self, queue, sdp_line, uid=None, need_stop=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uid = uid
        self.queue = queue
        self.sdp_line = sdp_line
        self.need_stop = need_stop

        self.pool = concurrent.futures.ThreadPoolExecutor(
            (os.cpu_count() or 1))
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logging.basicConfig(level=logging.INFO)
        self.is_running = False
        self._sdp_parse(sdp_line, 'video{}.sdp'.format(connections+1))
        self.video_gst = VideoGStream(sdp_line)

    def run(self):
        print("run until complete...")
        self.is_running = True
        self.loop.run_until_complete(self._recognize())
        self.loop.run_forever()

    @staticmethod
    def _sdp_parse(sdp_line, file_name='video.sdp'):

        line_res_sdp = sdp_line.replace('\r', ' ')
        file = open(file_name, "w")
        file.writelines((line_res_sdp))
        file.close()
        logging.info('Sdp file created: %s', file_name)

    @staticmethod
    def _process_chunk(rec, message):
        if isinstance(message, str) and message == '{"eof" : 1}':
            return "END", True

        prediction, _ = rec.update(message)
        return prediction, False

    async def _recognize(self):
        from gesture_recognizer import GestureRecognizer
        self.model = GestureRecognizer(os.path.join('DDNet_model.tflite'))
        self.video_gst.start()

        while self.is_running:
            response = f"Message from gesture module, but gstreamer appsink is None"
            if self.video_gst.appsink:
                sample = self.video_gst.get_sample()
                frame = self.video_gst.sample_to_array(sample)
                response = f"Message from gesture module, but input frame is None"
                if frame is not None:
                    response, stop = await self.loop.run_in_executor(self.pool, self._process_chunk, self.model, frame.copy())

            if not self.queue.full() and last_gesture_id != response['predicted_id']:
                last_gesture_id = response['predicted_id']
                self.queue.put(json.dumps(response))

            if self.need_stop:
                break
