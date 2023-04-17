#!/usr/bin/env python3

import json
import os
import sys
import uuid
import time
import asyncio
import websockets
import concurrent.futures
import logging
from multiprocessing import Process, Queue
from stt_mod import SttRecognizer
from gr_mod import GrRecognizer
import nest_asyncio
nest_asyncio.apply()

# TODO Change global vars...
PORTS = list(range(2000, 2500))
COMMANDS = ['stopAll', 'stopAudio', 'startAudio', 'stopVideo', 'startVideo']
MODELS = {}


class WebsocketRecServer(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = type('', (), {})()
        self.receiver = Queue(maxsize=10)
        self.pool = concurrent.futures.ThreadPoolExecutor(
            (os.cpu_count() or 1))
        self.loop = asyncio.get_event_loop()

    def run(self, server_interface='0.0.0.0', port=2700):
        logging.basicConfig(level=logging.INFO)
        self.args.interface = os.environ.get('server_interface', server_interface)
        self.args.port = int(os.environ.get('port', port))
        if len(sys.argv) > 1:
            self.args.model_path = sys.argv[1]
        start_server = websockets.serve(self._recognize, self.args.interface, self.args.port)
        logging.info("WS SERVER LISTENING ON %s:%d", self.args.interface, self.args.port)
        logging.info("*"*70)
        self.loop.run_until_complete(start_server)
        self.loop.run_forever()

    async def _recognize(self, websocket, path):
        logging.info('Connection from %s', websocket.remote_address)
        video_port = PORTS.pop()
        audio_port = PORTS.pop()
        await websocket.send(
            self._response_start(self.args.interface, video_port, audio_port, uuid.uuid1().urn[9:]))
        time.sleep(2)
        while True:
            message = await websocket.recv()
            if message == 'END':
                break
            response = json.loads(message)
            time.sleep(0.05)
            await self._from_receiver_data(ws=websocket)
            self._execute_command(response)

    async def _from_receiver_data(self, ws):
        if not self.receiver.empty():
            response = self.receiver.get(timeout=1)
            await ws.send(response)
        else:
            await ws.send("queue is empty")

    def _execute_command(self, response):
        # TODO check how will create ID ???
        ident = response.get('id')
        language = response.get('lang')
        command = response.get('command')
        sdp = response.get('data')['sdp']
        if command in COMMANDS:
            if command == 'stopAll':
                logging.info('STOP ALL command')
                for model in MODELS.values():
                    self.receiver.close()
                    if model.is_alive():
                        model.kill()
            if command == 'startAudio':
                logging.info(f'START {language} command')
                if ident not in MODELS.keys():
                    MODELS[ident] = SttRecognizer(queue=self.receiver, lang=language, sdp_line=sdp, uid=ident)
                    MODELS[ident].start()
                else:
                    logging.info(f'STARTing {language} failed, id_{ident} already exists')
            if command == 'stopAudio':
                logging.info(f'STOP {language} command')
                if ident in MODELS.keys():
                    if MODELS[ident].is_alive():
                        MODELS[ident].kill()
                    else:
                        logging.info(f'STOPing {language} failed, id_{ident} is not alive')
                else:
                    logging.info(f'STOPing {language} failed, id_{ident} is not exists')
            if command == 'startVideo':
                logging.info(f'START gr command')
                if ident not in MODELS.keys():
                    MODELS[ident] = GrRecognizer(queue=self.receiver, sdp_line=sdp)
                    MODELS[ident].start()
                else:
                    logging.info(f'STARTing gr failed, id_{ident} already exists')
            elif command == 'stopVideo':
                logging.info(f'STOP gr command')
                if ident in MODELS.keys():
                    if MODELS[ident].is_alive():
                        MODELS[ident].kill()
                    else:
                        logging.info(f'STOPing gr failed, id_{ident} is not alive')
                else:
                    logging.info(f'STOPing gr failed, id_{ident} is not exists')

    @staticmethod
    def _response_start(remote_address, video_port, audio_port, session):
        dct = {'result': "success", 'address': remote_address, 'videoPort': video_port, 'audioPort': audio_port,
               'session': session}
        return json.dumps(dct)


if __name__ == '__main__':
    serv = WebsocketRecServer()
    serv.start()
