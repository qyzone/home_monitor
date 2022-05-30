# -*- coding: utf-8 -*-
# @FileName: mailsender.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit

import smtplib
from email.mime.text import MIMEText  # 负责构造文本
from email.mime.application import MIMEApplication  # 负责构造附件
from email.mime.image import MIMEImage  # 负责构造图片
from email.mime.multipart import MIMEMultipart  # 负责将多个对象集合起来
from email.header import Header
import yaml
from tools.htmlmaker import HtmlMaker
import time


class MailSender:
    init_path = "./tools/config/init_email.yaml"

    def __init__(self):
        with open(self.init_path, 'r', encoding='utf-8') as f_yaml:
            init = yaml.safe_load(f_yaml)
        auth = init['auth']
        conf = init['conf']
        # SMTP服务器,这里使用QQ邮箱
        self.mail_host = conf['smtpHost']
        self.account = auth['account']
        self.valid_code = auth['validCode']
        self.nickname = auth['nickname']

        self.smtp = smtplib.SMTP_SSL(self.mail_host, 465)
        # self.smtp.set_debuglevel(1)  # 打印SMTP服务器交互信息
        # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
        self.smtp.login(self.account, self.valid_code)

    def send(self, receivers: list, subject, content, attachments: [MIMEApplication] = None):
        mm = MIMEMultipart('related')
        # mm["From"] = self.sender_name
        mm["From"] = f'{self.nickname}<{self.account}>'
        # 构建接受者名称格式 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
        receivers_name = []
        for receiver in receivers:
            receiver_name = f'{receiver}<{receiver}>'  # 格式："sender<mail address>"
            receivers_name.append(receiver_name)
        mm["To"] = ",".join(receivers_name)  # "receiver_1_name<**@qq.com>,receiver_2_name<***@**.com>"
        mm["Subject"] = Header(subject, 'utf-8')
        # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
        text_msg = MIMEText(content, 'html', 'utf-8')
        # 添加html文本到邮件中
        mm.attach(text_msg)
        # 添加附件到邮件中
        if attachments:
            for _attachment in attachments:
                mm.attach(_attachment)
        # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
        self.smtp.sendmail(self.account, receivers, mm.as_string())
        print("邮件发送成功")

    def quit(self):
        self.smtp.quit()
        print("关闭SMTP对象")


if __name__ == '__main__':
    # Example
    print(f"新建对象{time.time()}")
    mail_sender = MailSender()
    print(f"建成对象{time.time()}")
    # 收件人
    _receivers = ["finebit@qq.com"]
    # 邮件主题
    _subject = "测试内容"
    # 邮件正文内容
    html_maker = HtmlMaker()
    html_maker.add_content("span文字", "span")  # 添加span标签及内容
    img_name = "avatar.jpg"
    html_maker.add_content(img_name, "img")  # 添加图片标签
    # 附件 二进制读取图片
    print(f"读取图片{time.time()}")
    for i in range(5):
        with open(img_name, "rb") as fp:
            picture = MIMEImage(fp.read())
            picture.add_header('Content-ID', f'<{img_name}>')
    print(f"读完图片{time.time()}")
    mail_sender.send(_receivers, _subject, html_maker.html_msg, [picture])
    mail_sender.quit()
