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
            "讲解要点：左侧按真实 Edge Policy 32-step profiling 展示单次请求；"
            "默认 guidance=3、4 个 UniPC step，因此条件/无条件共 8 次 MoT 前向。"
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

    # Top I/O preparation chain
    chain_y = 242
    c.node(84, chain_y, 260, 88, "build_sample", "RGB 540×640×3 · state 8D\n→ video [3,33,540,640]", COLORS["soft_blue"], "8FB9ED", COLORS["blue"], "I/O")
    c.arrow(348, chain_y + 44, 388, chain_y + 44, COLORS["blue"], 4)
    c.node(392, chain_y, 310, 88, "ActionTransformPipeline", "resize/pad + action padding\nV [3,33,544,736] · A [33,64]", COLORS["soft_cyan"], "8CCED9", COLORS["cyan"], "T")
    c.arrow(706, chain_y + 44, 746, chain_y + 44, COLORS["cyan"], 4)
    c.node(750, chain_y, 378, 88, "build_batch", "增加 batch 维并整理 prompt\nV [1,3,33,544,736] · A [1,33,64]", COLORS["soft_green"], "90CFAD", COLORS["green"], "B")

    # prepare_data_total container
    py, ph = 356, 280
    c.rect(84, py, 1044, ph, "F9FBFE", "AFC2D9", 14, 2)
    c.badge(100, py + 15, 176, "prepare_data_total", COLORS["navy"])
    c.text(292, py + 18, 500, 28, "条件编码 · 序列打包 · 扩散初值", 18, COLORS["muted"], True)

    c.node(108, 430, 204, 104, "get_data_and_condition", "读取多视角 video / state\n建立条件字典", "FFFFFF", "AFC2D9", COLORS["navy"], None, 15)
    c.arrow(314, 456, 350, 414, COLORS["navy"], 3)
    c.arrow(314, 505, 350, 554, COLORS["navy"], 3)
    c.node(354, 391, 242, 102, "vae_encode", "[1,48,9,34,46]\n→ crop [1,48,9,33,40]", COLORS["soft_cyan"], "8CCED9", COLORS["cyan"], "VAE")
    c.node(354, 512, 242, 82, "tokenize_text", "cond 105→107 · uncond 17→19", COLORS["soft_blue"], "8FB9ED", COLORS["blue"], "TXT")
    c.arrow(600, 441, 636, 466, COLORS["blue"], 3)
    c.arrow(600, 560, 636, 525, COLORS["blue"], 3)
    c.node(640, 430, 232, 116, "initial_pack", "vision 3060 + action 33\ncond L=3200 · uncond L=3112\nhidden = 2048", COLORS["soft_amber"], "EDC881", COLORS["amber"], "PACK")
    c.node(896, 430, 208, 116, "initialize noise", "V 570,240 + A 2,112\ntimestep: V 2720 / A 32", COLORS["soft_red"], "E9A9B1", COLORS["red"], "ε")
    c.arrow(874, 488, 892, 488, COLORS["amber"], 3)
    c.text(102, 606, 984, 24, "关键：文本/视觉/动作被打成一个 packed sequence；条件与无条件 token 长度不同。", 16, COLORS["navy"], True)

    # sampler_total container
    sy, sh = 656, 238
    c.rect(84, sy, 1044, sh, "F9FBFE", "AFC2D9", 14, 2)
    c.badge(100, sy + 15, 142, "sampler_total", COLORS["navy"])
    c.badge(946, sy + 15, 158, "UniPC × 4 steps", COLORS["green"])
    c.text(260, sy + 18, 620, 28, "双分支去噪 → CFG 合流 → latent 更新", 18, COLORS["muted"], True)

    c.node(110, 716, 316, 68, "conditional_forward", "L=3200 → V̇ [1,48,9,33,40] · Ȧ [33,64]", COLORS["soft_blue"], "8FB9ED", COLORS["blue"])
    c.node(110, 800, 316, 66, "unconditional_forward", "L=3112 · 仅 guidance ≠ 1 启用", COLORS["soft_navy"], "AFC2D9", COLORS["navy"])
    c.arrow(430, 750, 500, 764, COLORS["blue"], 3)
    c.arrow(430, 833, 500, 804, COLORS["navy"], 3)
    diamond = c.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.DIAMOND, inch(504), inch(730), inch(152), inch(100))
    diamond.fill.solid(); diamond.fill.fore_color.rgb = rgb(COLORS["soft_amber"])
    diamond.line.color.rgb = rgb(COLORS["amber"]); diamond.line.width = Pt(1.2)
    c.draw.polygon([(580, 730), (656, 780), (580, 830), (504, 780)], fill=hex_tuple(COLORS["soft_amber"]), outline=hex_tuple(COLORS["amber"]))
    c.text(517, 748, 126, 64, "CFG velocity\nv_u + 3(v_c−v_u)", 16, COLORS["ink"], True, "center", "middle")
    c.arrow(660, 779, 704, 779, COLORS["amber"], 3)
    c.node(708, 724, 380, 110, "latent update", "UniPC 更新 vision/action noise\n循环进入下一 timestep", COLORS["soft_green"], "90CFAD", COLORS["green"], "LOOP")
    c.line(898, 838, 898, 876, COLORS["green"], 3)
    c.line(898, 876, 96, 876, COLORS["green"], 3)
    c.line(96, 876, 96, 750, COLORS["green"], 3)
    c.arrow(96, 750, 106, 750, COLORS["green"], 3)
    c.text(455, 848, 330, 22, "4 steps × 2 branches = 8 次 MoT forward", 15, COLORS["green"], True, "center")

    # Outputs
    c.arrow(330, 896, 330, 916, COLORS["green"], 3)
    c.node(108, 918, 442, 78, "action_postprocess", "[33,64] → [33,8] → drop state row → [32,8]", COLORS["soft_green"], "90CFAD", COLORS["green"], "OUT")
    c.arrow(554, 957, 604, 957, COLORS["green"], 4)
    c.rect(608, 928, 198, 58, COLORS["green"], COLORS["green"], 13, 1)
    c.text(608, 928, 198, 58, "robot action\n[32, 8]", 19, "FFFFFF", True, "center", "middle")
    c.arrow(892, 896, 892, 922, COLORS["cyan"], 3, True)
    c.node(824, 924, 280, 70, "vae_decode（可选）", "仅 --decode-video；Edge 实测未记录", "FFFFFF", "8CCED9", COLORS["cyan"], "OPT")

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
