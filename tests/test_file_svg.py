import os

import aggdraw
import pytest

from PIL import Image, ImageColor, ImageDraw
from pillow_svg import SvgImagePlugin

from .helper import assert_image_equal, assert_image_similar

IMAGES_DIR = "tests/images"
SVG_DIR = "tests/svg"
use_aggdraw = False


def test_quick():
    ImageColor.colormap['currentcolor'] = ImageColor.colormap['black']

    for fn in ['rectangle']:
        im = Image.new('RGBA', (24, 24))
        if use_aggdraw:
            draw = aggdraw.Draw(im)
        else:
            draw = ImageDraw.Draw(im)
        SvgImagePlugin.draw_svg_file(os.path.join(SVG_DIR, fn+'.svg'), draw, im.size, '#00000000')
        compare = Image.open(os.path.join(IMAGES_DIR, fn+'.png'))
        debug = True
        if debug:
            # im.show()
            # compare.show()
            w, h = 24, 24
            ww, hh = w*2+3, h+5
            show_image = Image.new('RGBA', (ww, hh), 'cyan')
            show_image.paste(im, (1, 1, 1+w, 1+w))
            show_image.paste(compare, (2+w, 1, 2+w+w, 1+w))
            show_image.resize((ww*8, hh*8), resample=Image.BOX).show()

        assert im.size == (24, 24)
        assert_image_similar(im, compare, 1)

