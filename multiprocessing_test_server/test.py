#!/usr/bin/env python3

import asyncio
import websockets
import sys
import time
import json


async def run_test(uri):

    async with websockets.connect(uri) as websocket:

        print(await websocket.recv())
        resp = [json.dumps({"id": "2", "lang": "ru", "command": "startAudio", "data": {
                                       "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
           #  ,json.dumps({"id": "3", "lang": "gr", "command": "startVideo", "data": {
           #      "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
           # # ,json.dumps({"id": "1", "lang": "ar", "command": "stopAudio", "data": {
           #                         "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
           #  ,json.dumps({"id": "4", "lang": "gr", "command": "startVideo", "data": {
           #      "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
            # ,json.dumps({"id": "3", "lang": "gr", "command": "stopVideo", "data": {
            #     "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
            ,json.dumps({"id": "5", "lang": "ar", "command": "startAudio", "data": {
                                   "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}}),
            json.dumps({"id": "25", "lang": "ar", "command": "startAudio", "data": {
                                   "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
            ,json.dumps({"id": "5", "lang": "ar", "command": "stopAudio", "data": {
                                   "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})
            ,json.dumps({"id": "1", "lang": "ar", "command": "stopAll", "data": {
                                        "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=RTP Video\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 7245 RTP/AVP 102\r\na=rtpmap:102 H264/90000\r\na=recvonly"}})]
        await websocket.send(resp[0])
        s_t = time.time()
        i = 1
        while True:
            if time.time() - s_t > i*20:
                if i == len(resp):
                    await websocket.send(json.dumps({"END": 1}))
                    await asyncio.sleep(1)
                    break
                await websocket.send(resp[i])
                i += 1

            print(await websocket.recv())
            await websocket.send(resp[0])

s = time.time()

loop = asyncio.get_event_loop()
tasks = [loop.create_task(run_test('ws://localhost:2700'))
         for _ in range(int(sys.argv[1]))]
loop.run_until_complete(asyncio.wait(tasks))

en = time.time()
print(en - s)
