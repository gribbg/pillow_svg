from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFile
from typing import Type, Tuple, Dict, List, Union, cast
import math

from xml.etree.ElementTree import XMLParser
from functools import reduce

# font-family//font-weight//font-style -> path
font_mapping: Dict[str, str] = {}
svg_default_size = 480, 480
svg_default_background = "#FFFFFF"


def register_font(path: str, family: str, weight: str = "normal", style: str = "normal"):
    font_mapping["%s//%s//%s" % (family, weight, style)] = path
    pass


def set_svg_default(size: Tuple[int, int], background: str):
    svg_default_size = size
    svg_default_background = background


class SvgPainter:
    """
    Basic SVG support
    SVG 1.1 specification - https://www.w3.org/TR/SVG11/
    """

    svgns = "http://www.w3.org/2000/svg"
    finished = False

    def dump(self, obj):
        for attr in dir(obj):
            print("obj.%s = %r" % (attr, getattr(obj, attr)))

    def apply_transform(self, x, y, transforms: List[Tuple[str, List[str]]]):
        for (op, params) in reversed(transforms):
            if op == "scale":
                [x, y, zz] = self.apply_matrix(self.matrix(float(params[0]), 0.0, 0.0, float(params[1]) if len(params) > 1 else float(params[0]), 0.0, 0.0), [x, y, 1.0])
            if op == "translate":
                [x, y, zz] = self.apply_matrix(self.matrix(1.0, 0.0, 0.0, 1.0, float(params[0]), float(params[1]) if len(params) > 1 else 0), [x, y, 1.0])
            if op == "matrix":
                [x, y, zz] = self.apply_matrix(self.matrix(float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4]), float(params[5])), [x, y, 1.0])
            if op == "rotate":
                a = float(params[0])
                [x, y, zz] = self.apply_matrix(self.matrix(math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0), [x, y, 1.0])

            # print (op, params, x, y)

        return (x, y)

    def get_coords(self, x, y):
        x = float(x)
        y = float(y)

        for (tag, attrib) in reversed(self.tagstack):
            if "transform" in attrib:
                transforms = self.parse_transform(attrib["transform"])
                (x, y) = self.apply_transform(x, y, transforms)

        # project from viewBox
        x = (x - self.svgviewbox[0]) * (self.svgsize[0] / self.svgviewbox[2])
        y = (y - self.svgviewbox[1]) * (self.svgsize[1] / self.svgviewbox[3])

        return (x, y)

    def get_opacity(self):
        opacity = 1.0
        for (tag, attrib) in reversed(self.tagstack):
            if "opacity" in attrib:
                opacity = opacity * float(attrib["opacity"])
        return opacity

    def get_scale(self):
        return (self.svgsize[1] / self.svgviewbox[3])

    def get_attribute(self, key: str, default: str, inheritParent: bool):
        for (tag, attrib) in reversed(self.tagstack):
            if key in attrib:
                return attrib[key]
            if not inheritParent:
                return default

        return default

    def matrix(self, a: float, b: float, c: float, d: float, e: float, f: float):
        return [[a, c, e], [b, d, f], [0.0, 0.0, 1.0]]

    def apply_matrix(self, matrix, old):
        return [matrix[0][0] * old[0]+ matrix[0][1] * old[1] + matrix[0][2], 
            matrix[1][0] * old[0]+ matrix[1][1] * old[1] + matrix[1][2],
            1.0]

    def parse_transform(self, transform):
        transformList = [ ]
        xfs = transform.split(")")
        for xf in xfs:
            if xf.find("(") != -1:
                op = xf[:xf.index("(")]
                params = xf[xf.index("(")+1:].split(",")
                transformList.append((op, params))
        return transformList

    def apply_style(self, tag: str, attrib):
        def apply_style_to_attrib(style: str, attrib): # converts styles into attributes
            cssAttributes = style.split(";")
            for a in cssAttributes:
                if a.strip() != "":
                    (k, v) = a.split(":")
                    if k.strip() != "":
                        attrib[k.strip()] = v.strip()

        if "id" in attrib:
            id = attrib["id"]
            for (sel, style) in self.styles:
                if sel == "#" + id:
                    apply_style_to_attrib(style, attrib)

        if "class" in attrib:
            classes = attrib["class"].split(' ')
            for (sel, style) in self.styles:
                if sel in map(lambda cls: "." + cls, classes):
                    apply_style_to_attrib(style, attrib)

        if "style" in attrib:
            apply_style_to_attrib(attrib["style"], attrib)

    def parse_font_size(self, fontSize):
        return self.parse_size(fontSize)

    def parse_size(self, size):
        scale = self.get_scale()
        if type(size) == str:
            return int(round(float(size.replace("px", "")) * scale))
        return int(round(float(size) * scale))

    def get_ns_tag(self, tag: str) -> Tuple[str, str]:
        ns = tag[tag.index("{")+1:tag.index("}")]
        tag = tag[tag.index("}")+1:]
        return (ns, tag)

    def parse_color(self, colour: str, fill_opacity: float = 1.0):
        if colour == "none" or colour == None:
            return None
        col = ImageColor.getrgb(colour)
        # print(colour, col)
        if len(col) > 3: fill_opacity = (col[3] / 255.0)
        col = (col[0], col[1], col[2], int(self.get_opacity() * fill_opacity * 255))

        if self.draw.mode == "1": # bit mode
            bw_color = (col[0] + col[1] + col[2]) / (255 * 3)
            return int(bw_color)

        return col

    def resolve_font(self):
        default_font = "Arial"
        family = self.get_attribute("font-family", default_font, True).replace("'", "").replace("\"", "")
        weight = self.get_attribute("font-weight", "normal", True)
        style = self.get_attribute("font-style", "normal", True)
        size = self.parse_font_size(self.get_attribute("font-size", 16, True))
        font_key = "%s//%s//%s" % (family, weight, style)
        try:
            return ImageFont.truetype(font_mapping[font_key] if font_key in font_mapping else family, size)
        except:
            return ImageFont.truetype(default_font, size)

    def flush_text(self):
        text = self.text

        self.draw_text(text)
        # texts = text.split(" ")

        # first = True
        # for t in texts:
        #     if first:
        #         first = False
        #     else:
        #         self.draw_text(" ")
        #     self.draw_text(t)

    def draw_text(self, text: str):
        def draw_text_fix(xy: Tuple[float, float], text: str, font, fill):
            last_text_point = xy
            texts = text.split(" ")

            font_size = self.parse_font_size(self.get_attribute("font-size", 48, True))
            space_size = font_size / 3

            first = True
            for t in texts:
                if first:
                    first = False
                else:
                    last_text_point = (last_text_point[0] + space_size, last_text_point[1])
                size = self.draw.textsize(t, font = font)
                self.draw.text(last_text_point, t, font = font, fill = fill)
                last_text_point = (last_text_point[0] + size[0], last_text_point[1])

            return last_text_point

        def draw_text_size_fix(text: str, font: ImageFont):
            sizefull = self.draw.textsize(text, font = font)
            texts = text.split(" ")

            width = 0.0
            font_size = self.parse_font_size(self.get_attribute("font-size", 48, True))
            space_size = font_size / 3

            first = True
            for t in texts:
                if first:
                    first = False
                else:
                    width += space_size
                size = self.draw.textsize(t, font = font)
                width += size[0]

            return (width, sizefull[1])

        #text = self.textStack[-1]
        #self.textStack[-1] = ""
        self.text = ""
        (tag, attrib) = self.tagstack[-1]
        if (tag == "text" or tag == "tspan"):
            if text != "":
                alignment = self.get_attribute("text-align", "start", True)
                alignment = self.get_attribute("text-anchor", alignment, True)
                # print (text, self.getCoords(self.getAttribute("x", 0, True), self.getAttribute("y", 0, True)), self.getAttribute("font-size", 0, True), self.parseFontSize(self.getAttribute("font-size", 0, True)))
                last_text_point = self.lasttextpoint
                font = self.resolve_font()
                ascent, descent = font.getmetrics()

                size = draw_text_size_fix(text, font)
                if alignment == "end":
                    draw_text_fix((last_text_point[0] - size[0], last_text_point[1] - ascent), text, font, self.parse_color(self.get_attribute("fill", "#000", False)))
                    self.lasttextpoint = (last_text_point[0], last_text_point[1])
                if alignment == "middle":
                    last_text_point = draw_text_fix((last_text_point[0] - size[0] / 2, last_text_point[1] - ascent), text, font, self.parse_color(self.get_attribute("fill", "#000", False)))
                    self.lasttextpoint = (last_text_point[0] + size[0] / 2, last_text_point[1])
                else:
                    draw_text_fix((last_text_point[0], last_text_point[1] - ascent), text, font, self.parse_color(self.get_attribute("fill", "#000", False)))
                    self.lasttextpoint = (last_text_point[0] + size[0], last_text_point[1])

    def parse_path_data(self, data: str):
        def func(result: List[List[str]], c: str):
            last = result[-1]
            if c.isalpha():
                result.append([c])
                result.append([])
            elif c == "-":
                result.append([c])
            elif c == " " or c == ",":
                result.append([])
            else:
                last.append(c)
            return result

        return list(filter(None, map(lambda x: "".join(x), reduce(func, data, [[]]))))

    def flush_line(self, pointList: List[float], width: int, stroke: Union[int, Tuple[int, int, int]], fill: Union[int, Tuple[int, int, int]]):
        if len(pointList) != 0:
            if (fill != None):
                self.draw.polygon(pointList, fill=fill)
            if (stroke != None):
                self.draw.line(pointList, width=width, fill=stroke)


    def draw_path(self, data: List[str]):
        def relative_absolute_point(point: Tuple[float, float], lastpoint: Tuple[float, float], op: str):
            relative = op[0].islower()
            if (not relative):
                return point
            return (point[0] + lastpoint[0], point[1] + lastpoint[1])

        def add_transform_to_point_list(point_list: List[float], point: Tuple[float, float]):
            transformed_point = self.get_coords(point[0], point[1])
            point_list.append(transformed_point[0])
            point_list.append(transformed_point[1])

        def draw_bezier(point_list: List[float], lastpoint: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p: Tuple[float, float]):
            transformed_last_point = self.get_coords(lastpoint[0], lastpoint[1])
            transformed_p = self.get_coords(p[0], p[1])
            max_range = int(max(abs(transformed_last_point[0] - transformed_p[0]), abs(transformed_last_point[1] - transformed_p[1])));
            # print(lastpoint, p1, p2, p)
            for i in range(0, max_range):
                t = i / float(max_range)
                ipointx = math.pow(1 - t, 3) * lastpoint[0] + 3 * math.pow(1 - t, 2) * t * (p1[0]) + 3 * (1 - t) * math.pow(t, 2) * (p2[0]) + math.pow(t, 3) * p[0]
                ipointy = math.pow(1 - t, 3) * lastpoint[1] + 3 * math.pow(1 - t, 2) * t * (p1[1]) + 3 * (1 - t) * math.pow(t, 2) * (p2[1]) + math.pow(t, 3) * p[1]
                add_transform_to_point_list(point_list, (ipointx, ipointy))
            add_transform_to_point_list(point_list, p)

        stroke_width = self.parse_size(self.get_attribute("stroke-width", 1.0, True))
        stroke = self.parse_color(self.get_attribute("stroke", None, True))
        fill = self.parse_color(self.get_attribute("fill", None, True))
        # print(stroke_width, stroke, fill)
        op = "M"
        opcount = 0
        x = 0
        origin = None
        last_point = (0.0, 0.0)
        last_p2 = (0.0, 0.0)
        point_list = [ ]
        while x < len(data):
            if data[x].isalpha():
                op = data[x]
                opcount = 0
                x += 1

            d: List[float] = list(map(lambda xx: xx if isinstance(xx, str) and xx.isalpha() else float(xx), data[x:x+6]))
            # print(op, d)

            param_len = 1
            if op in ["M", "m"] and opcount < 1:
                # Move
                self.flush_line(point_list, stroke_width, stroke, fill)
                point_list = [ ]
                last_point = relative_absolute_point((d[0], d[1]), last_point, op)
                if origin == None:
                    origin = last_point
                add_transform_to_point_list(point_list, last_point)
                param_len = 2
            elif op in ["L", "l", "M", "m"]:
                # Line
                last_point = relative_absolute_point((d[0], d[1]), last_point, op)
                add_transform_to_point_list(point_list, last_point)
                param_len = 2
            elif op in ["H", "h"]:
                # Horizontal move
                last_point = relative_absolute_point((d[0], last_point[1]), (last_point[0], 0), op)
                add_transform_to_point_list(point_list, last_point)
            elif op in ["V", "v"]:
                # Vertical move
                last_point = relative_absolute_point((last_point[0], d[0]), (0, last_point[1]), op)
                add_transform_to_point_list(point_list, last_point)
            elif op in ["C", "c", "S", "s"]:
                # Bezier curve
                short_hand = op in ["S", "s"]
                p1 = (last_point[0] - (last_p2[0] - last_point[0]), last_point[1] - (last_p2[1] - last_point[1]))
                p2 = relative_absolute_point((d[0], d[1]), last_point, op)
                p = relative_absolute_point((d[2], d[3]), last_point, op)
                if not short_hand:
                    p1 = relative_absolute_point((d[0], d[1]), last_point, op)
                    p2 = relative_absolute_point((d[2], d[3]), last_point, op)
                    p = relative_absolute_point((d[4], d[5]), last_point, op)
                draw_bezier(point_list, last_point, p1, p2, p)
                last_point = p
                last_p2 = p2
                param_len = 6 if not short_hand else 4
            elif op in ["Q", "q", "T", "t"]:
                # Quadratic bezier curve
                short_hand = op in ["T", "t"]
                # print("lastpoint", last_point, last_p2)
                p1 = (last_point[0] - (last_p2[0] - last_point[0]), last_point[1] - (last_p2[1] - last_point[1]))
                p = relative_absolute_point((d[0], d[1]), last_point, op)
                if not short_hand:
                    p1 = relative_absolute_point((d[0], d[1]), last_point, op)
                    p = relative_absolute_point((d[2], d[3]), last_point, op)
                # else:
                #     print("lastpoint", last_point)
                #     print("last_p2", last_point)
                #     print("p1", p1)
                #     print("p", p)
                draw_bezier(point_list, last_point, (p1[0] - (p1[0] - last_point[0])/3, p1[1] - (p1[1] - last_point[1])/3), (p1[0] - (p1[0] - p[0])/3, p1[1] - (p1[1] - p[1])/3), p)
                last_point = p
                last_p2 = p1
                param_len = 4 if not short_hand else 2
            elif op in ["z", "Z"]:
                # Close loop
                add_transform_to_point_list(point_list, origin)
                param_len = 0

            if not (op in ["C", "c", "S", "s", "Q", "q", "T", "t"]):
                last_p2 = last_point

            opcount += 1
            x += param_len

        self.flush_line(point_list, stroke_width, stroke, fill)

    def parse_style_tag(self):
        css = self.text
        #print(css)
        decls = css.split('}')
        for decl in decls:
            keyvalue = decl.split('{')
            if (len(keyvalue) == 2):
                self.styles.append((keyvalue[0].strip(), keyvalue[1].strip()))
        #print(self.styles)

    # open XML tag
    def start(self, tag: str, attrib):
        (ns, tag) = self.get_ns_tag(tag)
        if ns != self.svgns:
            return

        if self.draw and self.text != "":
            self.flush_text()

        self.tagstack.append((tag, attrib))

        if self.draw:
            self.apply_style(tag, attrib)

        if not self.draw and not tag == "svg":
            return

        if tag == "svg":
            self.svgsize = (float(self.get_attribute("width", self.imagesize[0], False)), float(self.get_attribute("height", self.imagesize[1], False)))
            if ("viewBox" in attrib):
                self.svgviewbox = list(map(lambda s: float(s.strip()), filter(None, attrib["viewBox"].split(" "))))
            else:
                self.svgviewbox = [0, 0, self.svgsize[0], self.svgsize[1]]
            #print self.svgViewBox

        if tag == "style":
            self.parse_style_tag()

        elif tag == "rect":
            #print self.tagStack[-1][1]
            (x1, y1) = self.get_coords(attrib["x"], attrib["y"])
            (x2, y2) = self.get_coords(float(attrib["x"]) + float(attrib["width"]), float(attrib["y"]))
            (x3, y3) = self.get_coords(float(attrib["x"]) + float(attrib["width"]), float(attrib["y"]) + float(attrib["height"]))
            (x4, y4) = self.get_coords(float(attrib["x"]), float(attrib["y"]) + float(attrib["height"]))

            if not isinstance(self.draw, ImageDraw.ImageDraw):
                from aggdraw import Brush, Pen, Draw
                draw = cast(Draw, self.draw)
                draw.setantialias(False)
                self.draw.polygon([x1, y1, x2, y2, x3, y3, x4, y4], Brush('black'))
                # self.draw.polygon([20, 20, 21, 20, 21, 21, 20, 21], Brush('green'))
                # self.draw.polygon([20, 0, 21, 0, 21, 1, 20, 1], Brush('blue'))
                # self.draw.polygon([22, 0, 22, 1, 23, 1, 23, 0], Brush('red'))
                draw.flush()
            else:
                self.draw.polygon([x1, y1, x2, y2, x3, y3, x4, y4],
                    outline = self.parse_color(self.get_attribute("stroke", "none", True), float(self.get_attribute("stroke-opacity", "1", False))),
                    fill = self.parse_color(self.get_attribute("fill", "none", True), float(self.get_attribute("fill-opacity", "1", False))))
                # self.draw.polygon([20, 20, 21, 20, 21, 21, 20, 21], fill='green')
                # self.draw.polygon([20, 0, 21, 0, 21, 1, 20, 1], fill='blue')
                # self.draw.polygon([22, 0, 22, 0, 22, 1, 22, 1], fill='red')

        elif tag == "ellipse" or tag == "circle":
            cc = 0.55191502449
            cx = float(attrib["cx"])
            cy = float(attrib["cy"])
            if tag == "ellipse":
                rx = float(attrib["rx"])
                ry = float(attrib["ry"])
            else:
                rx = float(attrib["r"])
                ry = float(attrib["r"])

            w = rx
            h = ry

            self.draw_path(["M", cx - w, cy, "C", cx - w, cy - cc * h, cx - cc * w, cy - h, cx, cy - h,
            "S", cx + w, cy - cc * h, cx + w, cy,
            "S", cx + cc * w, cy + h, cx, cy + h,
            "S", cx - w, cy + cc * h, cx - w, cy,
            "z"
            ])

        elif tag == "polyline":
            path_data = self.parse_path_data(self.get_attribute("points", "", False))
            self.draw_path(path_data)

        elif tag == "path":
            path_data = self.parse_path_data(self.get_attribute("d", "", False))
            self.draw_path(path_data)

        elif tag == "text" or tag == "tspan":
            if "x" in attrib:
                self.lasttextpoint = self.get_coords(self.get_attribute("x", 0, True), self.get_attribute("y", 0, True))

        self.depth += 1

    # close XML tag
    def end(self, tag):
        self.depth -= 1

        (ns, tag) = self.get_ns_tag(tag)

        if ns != self.svgns:
            return

        if self.draw:
            if tag == "style":
                self.parse_style_tag()
            if tag == "title" or tag == "desc":
                pass
            else:
                self.flush_text()

        #self.textStack.pop()

        self.tagstack.pop()

        if (len(self.tagstack) == 0):
            self.finished = True

    # text data
    def data(self, data):
        self.text = self.text + data
        # print(self.tagstack[-1][0], 'text', data)

    # finish parsing
    def close(self): pass

    def __init__(self, imagedraw: ImageDraw.ImageDraw, imagesize: Tuple[int, int], background: str):
        self.draw = imagedraw
        self.imagesize = imagesize
        self.depth = 0
        self.tagstack = [ ]
        self.textstack = [ ]
        self.lasttextpoint = [0, 0]
        self.text = ""
        self.svgsize = imagesize
        self.styles = [ ]

        # set default background colour
        if imagedraw and background:
            imagedraw.rectangle((0, 0, imagesize[0], imagesize[1]), self.parse_color(background))


class SvgDecoder(ImageFile.PyDecoder):
    """
    Python implementation of a SVG format decoder.
    """

    imagesize: Tuple[int, int] = None

    def init(self, args):
        (image_draw_mode, image, image_draw, image_size, background) = args
        image_draw_mode = "RGBA" if image_draw_mode == None else image_draw_mode
        if image != None: self.setimage(image)
        image_draw = ImageDraw.Draw(self.im, image_draw_mode) if image_draw == None else image_draw
        self.imagesize = (self.im.width, self.im.height) if image_size == None else image_size
        self.target = SvgPainter(image_draw, self.imagesize, background)
        self.parser = XMLParser(target=self.target)

    def decode(self, buffer):
        self.parser.feed(buffer)
        return (len(buffer) if not self.target.finished else -1, 0)

    def cleanup(self):
        self.parser.close()


Image.register_decoder("SVGXML", lambda mode, *args: SvgDecoder(mode, *args))


def draw_svg_file(file: str, imagedraw: ImageDraw.ImageDraw, imagesize: Tuple[int, int], background: str):
    decoder = SvgDecoder("", None, None, imagedraw, imagesize, background)
    f = open(file, "r")
    xmlString = f.read()
    decoder.decode(xmlString)
    decoder.cleanup()


def _accept(prefix):
    prefixStr = prefix.decode("utf-8")
    return "<?xml" in prefixStr or "<svg " in prefixStr or "<!DOC" in prefixStr


class SvgImageFile(ImageFile.ImageFile):
    """
    Python implementation of a SVG Rasterised Image.
    """

    format = "SVG"
    format_description = "SVG image"

    def _open(self):
        header = self.fp.read(4096)

        # size in pixels (width, height)
        self._size = svg_default_size
        try:
            # try to find the SVG width and height
            target = SvgPainter(None, self._size, None)
            parser = XMLParser(target=target)
            parser.feed(header)
            if target.svgsize:
                self._size = (int(target.svgsize[0]), int(target.svgsize[1]))
            parser.close()
        except:
            pass

        # mode setting
        self.mode = "RGB"

        # data descriptor
        self.tile = [("SVGXML", (0, 0) + self.size, 0, (None, self, None, None, svg_default_background))]


