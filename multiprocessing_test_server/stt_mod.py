#!/usr/bin/env python3

import nest_asyncio
import json
import os
import time
import asyncio
import concurrent.futures
import logging
from multiprocessing import Process
from vosk import Model, KaldiRecognizer

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GLib, GstApp

nest_asyncio.apply()

MODELS = {'ru': './models/model_ru',
          'ar': './models/model_ar',
          'en': './models/model_en'}


class GStreamer(object):
    def __init__(self, command):
        self.is_running = False
        self.idx = 0
        self.pipeline = Gst.parse_launch(command)

    def stop(self):
        if self.is_running:
            self.is_running = False
            self.pipeline.set_state(Gst.State.NULL)
            self.main_loop.quit()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.idx = 0

            self.main_loop = GLib.MainLoop()
            self.pipeline.set_state(Gst.State.PLAYING)
            # "appsink0" for example =)
            self.appsink = self.pipeline.get_by_name("audio")

    def get_sample(self):
        sample = None
        if self.is_running:
            sample = self.appsink.pull_sample()

            if sample is not None:
                self.idx += 1
        return sample



class SttRecognizer(Process):
    def __init__(self, queue, lang, sdp_line, phrase_list=None, sample_rate=16000, show_words=True,
                 max_alternatives=0, uid=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uid = uid
        self.lang = lang
        self.sdp_line = sdp_line
        self.queue = queue
        self.phrase_list = phrase_list
        self.sample_rate = sample_rate
        self.show_words = show_words
        self.max_alternatives = max_alternatives
        self.pool = concurrent.futures.ThreadPoolExecutor(
            (os.cpu_count() or 1))
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logging.basicConfig(level=logging.INFO)

    def run(self):
        print("run until complete...")
        self.model = Model(MODELS[self.lang])
        self.loop.run_until_complete(self._recognize())
        self.loop.run_forever()

    @staticmethod
    def _sdp_parse(sdp_line, file_name='audio.sdp'):
        line_res_sdp = sdp_line.replace('\r', ' ')
        file = open(file_name, "w")
        file.writelines((line_res_sdp))
        file.close()
        logging.info('Sdp file created: %s', file_name)

    @staticmethod
    def _process_chunk(rec, message):
        time.sleep(0.1)
        if message == '{"eof" : 1}':
            return rec.FinalResult(), True
        elif rec.AcceptWaveform(message):
            return rec.Result(), False
        else:
            return rec.PartialResult(), False

    async def _recognize(self):
        rec = None
        phrase_list = None
        self._sdp_parse(self.sdp_line, f'{self.uid}_audio.sdp')
        command = f"filesrc location={self.uid}_audio.sdp ! sdpdemux ! decodebin ! audioconvert !" \
                 f" audio/x-raw,format=S16LE,rate=16000,layout=interleaved,channels=2 ! appsink name=audio"
        self.gstreamer = GStreamer(command)
        self.video_gst.start()
        # while self.video_gst.is_running:
        while True:
            sample = self.video_gst.get_sample()
            buffer = sample.get_buffer()
            (result, mapinfo) = buffer.map(Gst.MapFlags.READ)
            data = mapinfo.data
            if isinstance(message, str) and 'config' in message:
                jobj = json.loads(message)['config']
                logging.info("Config %s", jobj)
                if 'phrase_list' in jobj:
                    phrase_list = jobj['phrase_list']
                if 'sample_rate' in jobj:
                    sample_rate = float(jobj['sample_rate'])
                if 'words' in jobj:
                    show_words = bool(jobj['words'])
                if 'max_alternatives' in jobj:
                    max_alternatives = int(jobj['max_alternatives'])
                continue
            if not rec:
                if phrase_list:
                     rec = KaldiRecognizer(self.model, self.sample_rate, json.dumps(self.phrase_list, ensure_ascii=False))
                else:
                     rec = KaldiRecognizer(self.model, self.sample_rate)
                rec.SetWords(self.show_words)
                rec.SetMaxAlternatives(self.max_alternatives)
            response, stop = await self.loop.run_in_executor(self.pool, self._process_chunk, rec, data)
            msg = f"Message from {self.lang} model"
            time.sleep(0.25)
            res = json.dumps({'id': self.uid, f'{self.lang}_recognizer': response})
            if not self.queue.full():
                self.queue.put(res)
