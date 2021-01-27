# -*- coding: utf-8 -*-
import time
import random
import os.path
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

import cv2


def tag_picture(imagepath, name):
    image = cv2.imread(imagepath, 1)
    cv2.imwrite('./result/%s_%s.png' % (name, str(time.time())[:10]), image)


# 定义端口为9100
define("port", default=9100, help="run on the given port", type=int)


def get_image(dir):
    files = os.listdir(dir)
    return random.choice(files)


class ImageHandler(tornado.web.RequestHandler):

    # get函数
    def get(self):
        dir = './static/images'
        img_src = get_image(dir)
        self.render('index.html', img_src=img_src, imgname=img_src)

    # post函数
    def post(self):
        filename = self.get_argument('rename')
        imgname = self.get_argument('imgname')
        imagepath = os.path.dirname(__file__)+'/static/images/%s' % imgname
        # print(filename)
        # print(imagepath)
        tag_picture(imagepath, filename)
        os.remove(imagepath)
        print(len(os.listdir('./static/images')))

        dir = './static/images'
        img_src = get_image(dir)
        self.render('index.html', img_src=img_src, imgname=img_src)


# 主函数
if __name__ == '__main__':
    # 开启tornado服务
    tornado.options.parse_command_line()
    # 定义app
    app = tornado.web.Application(
            handlers=[(r'/index', ImageHandler)
                      ],    # 网页路径控制
            template_path=os.path.join(os.path.dirname(__file__), "templates"),     # 模板路径
            static_path=os.path.join(os.path.dirname(__file__), "static"),          # 配置静态文件路径
          )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
