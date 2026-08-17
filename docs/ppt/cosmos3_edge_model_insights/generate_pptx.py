#!/usr/bin/env python3
"""Generate the editable Cosmos3-Edge analysis slide and its PNG preview.

Dependencies are intentionally not vendored. Run with python-pptx and Pillow on
PYTHONPATH; the generated PPTX uses native PowerPoint shapes and connectors.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
PPTX_OUT = ROOT / "cosmos3_edge_model_insights_page1.pptx"
PNG_OUT = ROOT / "cosmos3_edge_model_insights_page1.png"
WIDTH, HEIGHT = 1920, 1080
PX_PER_INCH = 144
FONT_FILE = Path(os.environ.get("COSMOS_PPT_FONT", "/tmp/NotoSansCJKsc-Regular.otf"))


COLORS = {
    "canvas": "F5F7FB",
    "panel": "FFFFFF",
    "ink": "14213D",
    "muted": "5E6B82",
    "hair": "D9E2EF",
    "grid": "EAF0F7",
    "navy": "183B66",
    "blue": "1677FF",
    "cyan": "09A9C8",
    "green": "20A464",
    "amber": "E69A24",
    "red": "D94A5A",
    "soft_blue": "EAF3FF",
    "soft_cyan": "E7F8FB",
    "soft_green": "EAF8F1",
    "soft_amber": "FFF5E4",
    "soft_red": "FDECEF",
    "soft_navy": "EDF2F8",
}


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def inch(px: float):
    return Inches(px / PX_PER_INCH)


def ppt_font_size(px: float):
    return Pt(px / 2)


def font(px: int, *, bold: bool = False):
    path = FONT_FILE if FONT_FILE.exists() else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), px, layout_engine=ImageFont.Layout.RAQM)


def hex_tuple(hex_color: str):
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class SlideCanvas:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333333)
        self.prs.slide_height = Inches(7.5)
        self.slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.image = Image.new("RGB", (WIDTH, HEIGHT), hex_tuple(COLORS["canvas"]))
        self.draw = ImageDraw.Draw(self.image)
        self._set_background()

    def _set_background(self):
        bg = self.slide.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(COLORS["canvas"])
        for x in range(0, WIDTH, 48):
            self.line(x, 0, x, HEIGHT, COLORS["grid"], 1)
        for y in range(0, HEIGHT, 48):
            self.line(0, y, WIDTH, y, COLORS["grid"], 1)

    def rect(self, x, y, w, h, fill, stroke="D9E2EF", radius=14, width=2):
        shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
        shape = self.slide.shapes.add_shape(shape_type, inch(x), inch(y), inch(w), inch(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.color.rgb = rgb(stroke)
        shape.line.width = Pt(max(0.5, width / 2))
        self.draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=hex_tuple(fill), outline=hex_tuple(stroke), width=width)
        return shape

    def line(self, x1, y1, x2, y2, color, width=2, dash=False):
        connector = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, inch(x1), inch(y1), inch(x2), inch(y2))
        connector.line.color.rgb = rgb(color)
        connector.line.width = Pt(max(0.5, width / 2))
        if dash:
            connector.line.dash_style = 4
            self.draw.line((x1, y1, x2, y2), fill=hex_tuple(color), width=width)
        else:
            self.draw.line((x1, y1, x2, y2), fill=hex_tuple(color), width=width)
        return connector

    def arrow(self, x1, y1, x2, y2, color="1677FF", width=4, dash=False):
        self.line(x1, y1, x2, y2, color, width, dash)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 12
        left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
        right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
        tri = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ISOSCELES_TRIANGLE, inch(x2 - 9), inch(y2 - 9), inch(18), inch(18))
        tri.rotation = math.degrees(angle) + 90
        tri.fill.solid()
        tri.fill.fore_color.rgb = rgb(color)
        tri.line.fill.background()
        self.draw.polygon([(x2, y2), left, right], fill=hex_tuple(color))

    def path_arrow(self, points, color="1677FF", width=4, dash=False):
        """Draw an editable orthogonal/segmented connector with one arrowhead."""
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            self.line(x1, y1, x2, y2, color, width, dash)
        (x1, y1), (x2, y2) = points[-2], points[-1]
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 12
        left = (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6))
        right = (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6))
        tri = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ISOSCELES_TRIANGLE, inch(x2 - 9), inch(y2 - 9), inch(18), inch(18))
        tri.rotation = math.degrees(angle) + 90
        tri.fill.solid()
        tri.fill.fore_color.rgb = rgb(color)
        tri.line.fill.background()
        self.draw.polygon([(x2, y2), left, right], fill=hex_tuple(color))

    def text(self, x, y, w, h, value, size=24, color="14213D", bold=False, align="left", valign="top", margin=0, line_spacing=1.0):
        box = self.slide.shapes.add_textbox(inch(x), inch(y), inch(w), inch(h))
        frame = box.text_frame
        frame.clear()
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = inch(margin)
        frame.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[valign]
        p = frame.paragraphs[0]
        p.text = value
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        p.font.name = "Microsoft YaHei"
        p.font.size = ppt_font_size(size)
        p.font.bold = bold
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(0)
        p.line_spacing = line_spacing

        pil_font = font(size, bold=bold)
        anchor = {"left": "la", "center": "ma", "right": "ra"}[align]
        tx = {"left": x + margin, "center": x + w / 2, "right": x + w - margin}[align]
        bbox = self.draw.multiline_textbbox((0, 0), value, font=pil_font, spacing=max(2, int(size * 0.25)), align=align)
        text_h = bbox[3] - bbox[1]
        ty = {"top": y + margin, "middle": y + (h - text_h) / 2, "bottom": y + h - text_h - margin}[valign]
        self.draw.multiline_text((tx, ty), value, font=pil_font, fill=hex_tuple(color), anchor=anchor, spacing=max(2, int(size * 0.25)), align=align)
        return box

    def badge(self, x, y, w, label, fill, fg="FFFFFF"):
        self.rect(x, y, w, 34, fill, fill, 17, 1)
        self.text(x, y, w, 34, label, 17, fg, True, "center", "middle")

    def node(self, x, y, w, h, title, detail, fill="FFFFFF", stroke="9EB8D7", accent="1677FF", flag=None, title_size=20):
        self.rect(x, y, w, h, fill, stroke, 12, 2)
        self.rect(x, y, 7, h, accent, accent, 3, 1)
        self.text(x + 18, y + 11, w - 30, 26, title, title_size, COLORS["ink"], True)
        self.text(x + 18, y + 41, w - 30, h - 50, detail, 16, COLORS["muted"], False, "left", "top", 0, 0.9)
        if flag:
            self.badge(x + w - 54, y + 9, 42, flag, accent)

    def save(self):
        notes = self.slide.notes_slide.notes_text_frame
        notes.text = (
            "讲解要点：左侧严格按 STEP 1→4 阅读。先把服务侧预处理折叠成 video/action/prompt 三路输入；"
            "prepare_data_total 分别完成 VAE、文本 tokenize，并按 text→vision→action 顺序打包，同时初始化联合噪声；"
            "sampler_total 以同一 packed condition 和 joint latent 执行条件/无条件 MoT、CFG 合流与 UniPC 更新；"
            "默认 guidance=3、4 个 UniPC step，因此共 8 次 MoT 前向。最后拆成 action 和可选 video 两路输出。"
            "右侧参数采用 Cosmos 3 与 LingBot-VA 官方论文/模型卡口径。"
            "LingBot-VA 是实时因果双流路线，不宜仅凭参数量宣称吞吐优劣。"
        )
        props = self.prs.core_properties
        props.title = "NVIDIA Cosmos3-Edge 模型洞察与竞品分析"
        props.subject = "模型分析情况｜架构、参数与具身创新"
        props.author = "Cosmos Framework"
        self.prs.save(PPTX_OUT)
        self.image.save(PNG_OUT, optimize=True)


def build():
    c = SlideCanvas()

    # Header
    c.badge(56, 40, 86, "MODEL 01", COLORS["navy"])
    c.text(160, 34, 1180, 60, "NVIDIA Cosmos3-Edge｜模型洞察与竞品分析", 46, COLORS["ink"], True)
    c.text(160, 95, 1180, 32, "模型分析情况：单次推理架构 · 参数/计算结构 · 具身关键创新", 21, COLORS["muted"])
    c.text(1460, 45, 404, 32, "EDGE POLICY · DROID", 18, COLORS["blue"], True, "right")
    c.text(1460, 79, 404, 28, "32-step observed runtime", 16, COLORS["muted"], False, "right")
    c.line(56, 140, 1864, 140, COLORS["hair"], 2)

    # Left architecture panel
    lx, ly, lw, lh = 56, 166, 1128, 856
    c.rect(lx, ly, lw, lh, COLORS["panel"], COLORS["hair"], 18, 2)
    c.badge(lx + 26, ly + 22, 44, "01", COLORS["blue"])
    c.text(lx + 84, ly + 17, 760, 38, "Edge 单次推理框图", 28, COLORS["ink"], True)
    c.text(lx + 810, ly + 23, 286, 28, "去除 WebSocket I/O", 16, COLORS["muted"], False, "right")

    # Step 1: collapse all service-side preprocessing into one audience-level block.
    c.badge(84, 242, 88, "STEP 1", COLORS["navy"])
    c.text(184, 242, 190, 30, "输入汇聚", 20, COLORS["ink"], True)
    c.rect(84, 282, 216, 258, COLORS["soft_navy"], "AFC2D9", 14, 2)
    c.text(102, 296, 180, 30, "Input & Batch Prep", 20, COLORS["ink"], True, "center")
    c.text(102, 328, 180, 34, "build_sample / transform / batch", 12, COLORS["muted"], False, "center")
    input_lanes = [
        (374, "VIDEO", "[1,3,33,544,736]", COLORS["cyan"], COLORS["soft_cyan"]),
        (436, "ACTION / STATE", "[33,64] · row0=state", COLORS["amber"], COLORS["soft_amber"]),
        (498, "PROMPT", "cond / uncond text", COLORS["blue"], COLORS["soft_blue"]),
    ]
    for yy, label, shape, accent, fill in input_lanes:
        c.rect(102, yy, 180, 50, fill, accent, 9, 2)
        c.text(112, yy + 4, 160, 19, label, 13, accent, True)
        c.text(112, yy + 23, 160, 20, shape, 12, COLORS["muted"])

    # Step 2: preparation is a coherent stage with three modality lanes and one merge.
    c.rect(320, 236, 808, 334, "F9FBFE", "8FB0D3", 15, 3)
    c.badge(338, 252, 88, "STEP 2", COLORS["blue"])
    c.badge(438, 252, 176, "prepare_data_total", COLORS["navy"])
    c.text(630, 254, 466, 26, "模态编码 → 顺序打包 → 联合扩散初值", 17, COLORS["muted"], True)

    c.node(348, 314, 230, 108, "VAE Encoder", "video → latent\n[1,48,9,34,46] → [1,48,9,33,40]", COLORS["soft_cyan"], "8CCED9", COLORS["cyan"], "V")
    c.node(348, 454, 230, 82, "Tokenize Text", "cond 105→107 · uncond 17→19", COLORS["soft_blue"], "8FB9ED", COLORS["blue"], "T")

    # Explicit input data routes and shapes.
    c.arrow(302, 399, 344, 365, COLORS["cyan"], 4)
    c.path_arrow([(302, 461), (322, 461), (322, 548), (642, 548), (642, 496), (658, 496)], COLORS["amber"], 4)
    c.arrow(302, 523, 344, 495, COLORS["blue"], 4)
    c.text(414, 532, 206, 18, "ACTION LATENT [33,64]", 12, COLORS["amber"], True, "center")

    # Merge initial_pack and initialize-noise into one logical stage.
    c.rect(662, 304, 438, 232, COLORS["soft_amber"], "E8BE6A", 14, 3)
    c.badge(680, 320, 192, "PACK + NOISE INIT", COLORS["amber"])
    c.text(884, 321, 198, 28, "严格 token 顺序", 16, COLORS["ink"], True, "right")
    c.arrow(582, 366, 658, 376, COLORS["cyan"], 4)
    c.text(586, 342, 70, 20, "V-LATENT", 11, COLORS["cyan"], True, "center")
    c.arrow(582, 496, 658, 454, COLORS["blue"], 4)
    c.text(584, 476, 72, 20, "TEXT IDS", 11, COLORS["blue"], True, "center")

    # The standard Policy packer is text, then vision, then action.
    token_cells = [
        (682, 370, 112, "① TEXT", "C107 / U19", COLORS["blue"], COLORS["soft_blue"]),
        (798, 370, 144, "② VISION", "3060 patches", COLORS["cyan"], COLORS["soft_cyan"]),
        (946, 370, 132, "③ ACTION", "33 tokens", COLORS["green"], COLORS["soft_green"]),
    ]
    for xx, yy, ww, label, detail, accent, fill in token_cells:
        c.rect(xx, yy, ww, 60, fill, accent, 8, 2)
        c.text(xx, yy + 7, ww, 20, label, 13, accent, True, "center")
        c.text(xx, yy + 31, ww, 18, detail, 12, COLORS["muted"], False, "center")
    c.arrow(794, 400, 796, 400, COLORS["blue"], 2)
    c.arrow(942, 400, 944, 400, COLORS["cyan"], 2)
    c.text(682, 441, 396, 20, "packed hidden: cond [3200,2048] · uncond [3112,2048]", 13, COLORS["navy"], True, "center")
    c.rect(682, 470, 396, 48, COLORS["soft_red"], "E9A9B1", 8, 2)
    c.text(692, 474, 376, 18, "JOINT NOISE / LATENT", 12, COLORS["red"], True, "center")
    c.text(692, 494, 376, 18, "noise_v [1,48,9,33,40]  +  noise_a [33,64]  → flat [572352]", 12, COLORS["muted"], False, "center")

    # One unmistakable hand-off from preparation into the sampler.
    c.path_arrow([(880, 540), (880, 588), (100, 588), (100, 696), (112, 696)], COLORS["navy"], 5)
    c.rect(408, 573, 354, 30, COLORS["navy"], COLORS["navy"], 15, 1)
    c.text(408, 573, 354, 30, "PACKED CONDITIONS + JOINT LATENT", 13, "FFFFFF", True, "center", "middle")

    # Step 3: sampler_total is the main denoising system boundary.
    sy, sh = 610, 258
    c.rect(84, sy, 1044, sh, "F9FBFE", "8FB0D3", 15, 3)
    c.badge(112, sy + 15, 88, "STEP 3", COLORS["green"])
    c.badge(212, sy + 15, 142, "sampler_total", COLORS["navy"])
    c.text(372, sy + 18, 520, 26, "双分支 MoT 去噪 → CFG 合流 → joint latent 更新", 17, COLORS["muted"], True)
    c.badge(946, sy + 14, 158, "UniPC × 4 steps", COLORS["green"])

    c.node(116, 676, 314, 72, "Conditional MoT forward", "C [3200,2048] → v_c{video, action}", COLORS["soft_blue"], "8FB9ED", COLORS["blue"], "C", 17)
    c.node(116, 770, 314, 70, "Unconditional MoT forward", "U [3112,2048] → v_u · guidance≠1", COLORS["soft_navy"], "AFC2D9", COLORS["navy"], "U", 16)
    c.line(100, 696, 100, 805, COLORS["navy"], 4)
    c.arrow(100, 805, 112, 805, COLORS["navy"], 4)
    c.arrow(434, 712, 516, 738, COLORS["blue"], 4)
    c.arrow(434, 805, 516, 782, COLORS["navy"], 4)
    diamond = c.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.DIAMOND, inch(520), inch(710), inch(152), inch(104))
    diamond.fill.solid(); diamond.fill.fore_color.rgb = rgb(COLORS["soft_amber"])
    diamond.line.color.rgb = rgb(COLORS["amber"]); diamond.line.width = Pt(1.2)
    c.draw.polygon([(596, 710), (672, 762), (596, 814), (520, 762)], fill=hex_tuple(COLORS["soft_amber"]), outline=hex_tuple(COLORS["amber"]))
    c.text(534, 729, 124, 66, "CFG MERGE\nv = v_u + 3(v_c−v_u)", 13, COLORS["ink"], True, "center", "middle")
    c.arrow(676, 762, 716, 762, COLORS["amber"], 4)
    c.node(720, 696, 378, 126, "Joint latent update", "UniPC 同步更新两种生成状态\nvideo latent [1,48,9,33,40]\naction latent [33,64]", COLORS["soft_green"], "90CFAD", COLORS["green"], "Z", 18)

    # Loop is visually separated from the forward path.
    c.path_arrow([(908, 826), (908, 852), (98, 852), (98, 712), (112, 712)], COLORS["green"], 3)
    c.text(456, 832, 344, 20, "STEP 1→4 · 共 8 次 MoT forward", 14, COLORS["green"], True, "center")

    # Step 4: split the final joint state into action and video generation paths.
    c.rect(84, 890, 1044, 108, "F9FBFE", "AFC2D9", 14, 2)
    c.badge(102, 905, 88, "STEP 4", COLORS["amber"])
    c.text(204, 906, 150, 26, "生成输出", 18, COLORS["ink"], True)
    c.path_arrow([(966, 826), (966, 880), (424, 880), (424, 916)], COLORS["green"], 4)
    c.path_arrow([(966, 880), (824, 880), (824, 916)], COLORS["cyan"], 4, True)
    c.rect(358, 920, 360, 60, COLORS["soft_green"], "90CFAD", 11, 2)
    c.text(372, 926, 332, 21, "ACTION GENERATION", 14, COLORS["green"], True)
    c.text(372, 949, 332, 22, "[33,64] → postprocess → robot action [32,8]", 13, COLORS["muted"])
    c.rect(738, 920, 366, 60, COLORS["soft_cyan"], "8CCED9", 11, 2)
    c.text(752, 926, 338, 21, "VIDEO GENERATION（可选）", 14, COLORS["cyan"], True)
    c.text(752, 949, 338, 22, "[1,48,9,33,40] → VAE Decoder → future video", 13, COLORS["muted"])

    # Right comparison panel
    rx, rw = 1208, 656
    c.rect(rx, 166, rw, 422, COLORS["panel"], COLORS["hair"], 18, 2)
    c.badge(rx + 24, 188, 44, "02", COLORS["amber"])
    c.text(rx + 82, 182, 500, 38, "参数与计算结构", 27, COLORS["ink"], True)
    c.text(rx + 24, 229, 606, 24, "采用官方口径；FLOPs 未统一披露，以 denoiser 次数作复杂度 proxy", 15, COLORS["muted"])

    tx, ty = rx + 24, 270
    col = [150, 146, 146, 166]
    headers = ["指标", "Cosmos3 Edge", "Cosmos3 Nano", "LingBot-VA"]
    fills = [COLORS["navy"], COLORS["blue"], COLORS["cyan"], COLORS["green"]]
    xx = tx
    for i, (label, ww) in enumerate(zip(headers, col)):
        c.rect(xx, ty, ww, 44, fills[i], fills[i], 0, 1)
        c.text(xx + 5, ty, ww - 10, 44, label, 15, "FFFFFF", True, "center", "middle")
        xx += ww
    rows = [
        ("总参 / 激活", "4B / dense 2B", "16B / dense 8B", "5.3B\n(5B+~350M)"),
        ("层数 · hidden", "28 · 2048", "36 · 4096", "30 · 3072/768"),
        ("FFN / 双流", "9216 · MoT", "12288 · MoT", "隔离 FFN 双流"),
        ("动作 / 去噪", "32×8 · 4 steps", "32-step profile*", "K=4 · V3/A10"),
        ("请求复杂度", "8 MoT forwards", "8 MoT forwards*", "异步因果双时标"),
    ]
    yy = ty + 44
    for ri, row in enumerate(rows):
        xx = tx
        h = 49
        for ci, (value, ww) in enumerate(zip(row, col)):
            fill = "F6F9FD" if ri % 2 == 0 else "FFFFFF"
            if ci == 0:
                fill = "EDF2F8"
            c.rect(xx, yy, ww, h, fill, COLORS["hair"], 0, 1)
            c.text(xx + 6, yy, ww - 12, h, value, 14 if ci else 15, COLORS["ink"] if ci == 0 else COLORS["muted"], ci == 0, "center", "middle")
            xx += ww
        yy += h
    c.text(tx, 562, 610, 18, "* Nano 为同一 profiling 配置口径；架构与 action chunk 是独立配置。", 13, COLORS["muted"])

    # Right innovation panel
    c.rect(rx, 610, rw, 412, COLORS["panel"], COLORS["hair"], 18, 2)
    c.badge(rx + 24, 632, 44, "03", COLORS["green"])
    c.text(rx + 82, 626, 500, 38, "具身关键创新与竞品定位", 27, COLORS["ink"], True)

    card_y = 680
    cards = [
        ("同体双塔 MoT", "统一 foundation model\n覆盖 reasoning / generation\n/ action 三种模式", COLORS["blue"], COLORS["soft_blue"]),
        ("Action-native Head", "动作专属 I/O projection\n与 FFN；状态行和未来 action\n同序列扩散", COLORS["amber"], COLORS["soft_amber"]),
        ("视频-动作联合预测", "future visual 作为辅助监督\n部署时可跳过 decoder\n降低端到端延迟", COLORS["green"], COLORS["soft_green"]),
    ]
    x = rx + 24
    for title, detail, accent, fill in cards:
        c.rect(x, card_y, 194, 142, fill, accent, 12, 2)
        c.rect(x, card_y, 194, 7, accent, accent, 3, 1)
        c.text(x + 14, card_y + 18, 166, 28, title, 18, COLORS["ink"], True, "center")
        c.text(x + 12, card_y + 52, 170, 80, detail, 12, COLORS["muted"], False, "center", "middle")
        x += 206

    c.rect(rx + 24, 842, 608, 118, COLORS["soft_navy"], "AFC2D9", 12, 2)
    c.badge(rx + 40, 857, 136, "vs LingBot-VA", COLORS["navy"])
    c.text(rx + 190, 850, 420, 46, "优势在“统一模型迁移闭环”", 20, COLORS["navy"], True, "left", "middle")
    c.text(rx + 40, 900, 564, 48, "Edge：同模型贯通理解→生成→动作；LingBot-VA：面向实时控制的因果双流。\n结论：能力边界更统一 ≠ 已证明实时吞吐更高。", 15, COLORS["muted"], False, "left", "middle")

    # Footer
    c.text(56, 1037, 1808, 22, "Sources: Cosmos 3 paper (arXiv:2606.02800v4) · NVIDIA Edge Policy model card · LingBot-VA paper/repo · local Edge/Nano profiling", 13, COLORS["muted"], False, "left")
    c.text(1720, 1037, 144, 22, "2026.08", 13, COLORS["muted"], True, "right")

    c.save()
    print(PPTX_OUT)
    print(PNG_OUT)


if __name__ == "__main__":
    build()
