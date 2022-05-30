# -*- coding: utf-8 -*-
# @FileName: videoreceiver.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit
# Desc: 萤石云视频流对接
import time
import requests
import yaml

# 视频配置
HEADERS = {'Host': 'open.ys7.com', 'Content-Type': 'application/x-www-form-urlencoded'}


def date2stamp(datetime):
    time_arr = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_arr)
    return timestamp


class VideoReceiver:
    temp_path = "./tools/config/temp.yaml"
    init_path = "./tools/config/init_video.yaml"

    def __init__(self, remote=False):
        self.remote = remote  # 默认读取本地视频流
        with open(self.temp_path, 'r', encoding='utf-8') as fp:
            self.temp = yaml.safe_load(fp)
        if remote:
            with open(self.init_path, 'r', encoding='utf-8') as f_yaml:
                init = yaml.safe_load(f_yaml)
            self.conf = init['conf']
            auth = init['auth']
            self.app_key = auth['appKey']
            self.app_secret = auth['appSecret']
            self.device_serial = auth['deviceSerial']

    def update_yaml(self):
        with open(self.temp_path, 'w', encoding='utf-8') as fp:
            yaml.safe_dump(self.temp, fp)

    def update_access_token(self):
        token_url = "https://open.ys7.com/api/lapp/token/get"
        data = {
            'appKey': self.app_key,
            'appSecret': self.app_secret
        }
        response = requests.post(token_url, headers=HEADERS, params=data)
        print(response.json()['msg'])
        if response.json()['code'] == '200':
            access_token = response.json()['data']
            self.temp['accessToken'] = access_token
            self.update_yaml()

    def get_video_url(self):
        if not self.remote:
            return True, self.temp['local']['url']
        if time.time()*1000 > self.temp['accessToken']['expireTime']:
            self.update_access_token()
        if time.time() > date2stamp(self.temp['remote']['expireTime']):
            try:
                video_url = f"https://open.ys7.com/api/lapp/v2/live/address/get"
                data = {
                    'accessToken': self.temp['accessToken']['accessToken'],
                    'deviceSerial': self.device_serial,
                    'quality': self.conf['quality'],
                    'protocol': self.conf['protocol'],
                    'expireTime': self.conf['expireTime'],
                    'supportH265': self.conf['supportH265']
                }
                response = requests.post(video_url, params=data)
                msg = response.json()['msg']
                if "Operation succeeded" not in msg:
                    return False, msg
                self.temp['remote'] = response.json()['data']
                self.update_yaml()
            except Exception as e:
                return False, str(e)
        return True, self.temp['remote']['url']


if __name__ == '__main__':
    # Example
    v_receiver = VideoReceiver(remote=True)
    ret, url = v_receiver.get_video_url()
    if ret:
        print(url)
    else:
        print(f"Error:{url}")
