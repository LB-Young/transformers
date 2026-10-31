# -*- coding: utf-8 -*-
from PIL import Image
im = Image.open(r"E:\transformers\bayon-note\InternVL\internvl-3\_shots\InternVL-3-家族架构详解_desktop.jpg")
print("size:", im.size)
# 家族表位于页面上部，按比例裁取
w,h = im.size
crop = im.crop((0, int(h*0.075), w, int(h*0.135)))
crop.save(r"E:\transformers\bayon-note\InternVL\internvl-3\_shots\_check_family_crop.png")
print("saved")
