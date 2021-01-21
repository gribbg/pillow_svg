SVG Plugin for Pillow
#####################

A plugin to implement basic SVG read support here for ``Pillow``.
Currently supports basic paths, shapes, and text.

Usage
=====

Since ``SvgImagePlugin`` is not built-in to ``Pillow``, it must be imported
to activate it as a plugin.  This only has to happen once per process::

    import pillow_svg.SvgImagePlugin
    from PIL import Image

    svg_image = Image.open('example.svg')
    svg_image.show()


Installation
============

Install via pip::

    pip install pillow-svg

Clone the source::

    git clone https://github.com/loune/pillow_svg


Authors
=======

*   Loune Lam
*   Glenn Gribble
