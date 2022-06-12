# -*- coding: utf-8 -*-
# @FileName: predict.py
# @Project: home_monitor
# @Author: Finebit
from email.mime.image import MIMEImage
from pathlib import Path
from threading import Thread

from torchvision.transforms import transforms

from tools.dataloader import Resize
from tools.torch_tools import time_sync
from tools.videoreceiver import VideoReceiver
from tools.mailsender import MailSender
from tools.htmlmaker import HtmlMaker

import torch
from torch.nn import functional as nnf
import cv2

from models.focusnet import FocusNet as Net
# from models.cnn import Net


def send_mail(sub_images, save_dir, send=True):
    save_dir = Path(save_dir / 'images')
    save_dir.mkdir() if (not save_dir.exists()) else True
    mail_sender = MailSender()
    # 邮件主题
    subject = "监控信息提醒"
    # 邮件正文内容
    html_maker = HtmlMaker()
    html_maker.add_content("提醒类型: 出现人形", "span")  # 添加span标签及内容
    pictures = []
    for sub_image in sub_images:
        img = sub_image[0]
        image_name = sub_image[1]
        c = sub_image[2]
        print(c.values)
        save_path = save_dir / image_name
        cv2.imwrite(str(save_path), img)  # 保存文件
        html_maker.add_content(image_name, "img")  # 添加图片标签
        # 附件 二进制读取图片
        with open(str(save_path), "rb") as fp:
            picture = MIMEImage(fp.read())
        picture.add_header('Content-ID', f'<{image_name}>')
        pictures.append(picture)
    if send:
        mail_sender.send(subject, html_maker.html_msg, pictures)
    mail_sender.quit()


def save_video(images, save_dir):
    save_dir = Path(save_dir / 'videos')
    save_dir.mkdir() if (not save_dir.exists()) else True
    vid_name = str(int(time_sync()))
    save_path = (save_dir / vid_name).with_suffix('.mp4')
    fps, w, h = 30, images[0].shape[1], images[0].shape[0]
    vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for image in images:
        vid_writer.write(image)
    print(f"save {save_path} success!")


def predict(net, frame, device):
    resize = Resize((180, 320))  # h w
    frame = resize(frame)
    tn = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    frame = tn(frame)
    frame = torch.unsqueeze(frame, 0)
    frame = frame.to(device)
    output = net(frame)
    output = nnf.softmax(output, dim=-1)
    return output


@torch.no_grad()
def run(
        source,
        checkpoint_path,
        view_img=False,  # show results
):
    source = str(source)
    is_img = source.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))
    # Directories
    save_dir = Path("runs/predict")  # increment run
    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    net = Net()
    net.to(device)  # cuda内核占用1.4G内存
    model_pth = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(model_pth['state_dict'])
    net.eval()
    # load source
    if is_img:
        img = cv2.imread(source)
        output = predict(net, img, device)
        print(output)
        c = torch.max(output[0], 0)
        print(c)
    else:
        cap = cv2.VideoCapture(source)
        # parameter initial
        images = []
        sub_images = []
        wait_frames = 0
        save_img = False
        while cap.isOpened():
            ret, frame = cap.read()
            output = predict(net, frame, device)
            c = torch.max(output[0], 0)
            # send email and save video
            if c.indices == 1:
                if not save_img:
                    images = []  # 清空内容
                    sub_images = []
                    save_img = True
                images.append(frame)
                wait_frames = 0  # 防止误识别累计
                if len(images) % 10 == 1:  # images[]从1开始
                    sub_images.append([frame, f'{int(time_sync() * 10)}.jpg', c])
                    if len(sub_images) == 2:
                        print("send email")
                        thread = Thread(target=send_mail, args=(sub_images, save_dir, True))
                        thread.start()
                        pass
            elif save_img:
                wait_frames += 1
                images.append(frame)
                if wait_frames % 3 == 1:  # 用于收集误识别数据集, 可注释
                    cv2.imwrite(f'runs/predict/images/{int(time_sync() * 10)}_no.jpg', frame)
                # save video
                if wait_frames == 10:  # 防止未成功识别的情况，继续观察几帧
                    wait_frames = 0
                    thread = Thread(target=save_video, args=(images, save_dir))
                    thread.start()
                    save_img = False

            # Stream results
            if view_img:
                cv2.imshow(source, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 1 millisecond
                    break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    video_receiver = VideoReceiver()
    _ret, _source = video_receiver.get_video_url()
    if not _ret:
        print(_source)
        exit(0)
    # source = 'test.mp4'
    # source = 'person.jpg'  # 1
    # source = 'nothing.jpg'  # 0

    _checkpoint_path = f"runs/train/best.pth.tar"

    run(_source, _checkpoint_path, view_img=True)
