# -*- coding: utf-8 -*-
# @FileName: htmlmaker.py
# Version: 0.0.1
# @Project: home_monitor
# @Author: Finebit


def add_tag(content, tag='span'):
    if tag == 'img':
        img_name = content
        html_img = f'<img src="cid:{img_name}">'
        return add_tag(html_img, 'p')
    return f'<{tag}>{content}</{tag}>'


class HtmlMaker:
    html_msg: str = ""

    def add_content(self, content, tag):
        self.html_msg += add_tag(content, tag)


if __name__ == '__main__':
    a = "asdsfds.jpg"
    print(a.lower().endswith())

