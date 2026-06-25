"""PDF 的确定性处理：整页渲染、定位嵌入图坐标框、按坐标裁剪渲染。

关键约束（来自 _demo 实测）：配图一律「按坐标裁剪渲染」获取；
禁止用 Pixmap(doc, xref) 直接抠原始嵌入图对象（会得全黑图）。
"""

from __future__ import annotations

import fitz  # PyMuPDF

# 渲染缩放：2x ≈ 144dpi，清晰度足够喂 VLM / 存配图
DEFAULT_ZOOM = 2.0


def render_page_png(page: fitz.Page, zoom: float = DEFAULT_ZOOM) -> bytes:
    """整页渲染成 PNG 字节。"""
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes("png")


def list_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """返回页面上每个嵌入图的坐标框（页坐标系）。"""
    rects: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        for r in page.get_image_rects(xref):
            rects.append(r)
    return rects


def content_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """过滤掉页眉/页脚/logo 等装饰图，返回按 y 排序的「内容配图」框。

    启发式：跳过位于上 8% / 下 8% 页边的窄条，以及面积过小的图。
    """
    h = page.rect.height
    w = page.rect.width
    page_area = h * w
    top_margin = h * 0.08
    bottom_margin = h * 0.92

    kept: list[fitz.Rect] = []
    for r in list_image_rects(page):
        # 整框落在顶部/底部页边内 → 视为页眉/页脚装饰
        if r.y1 <= top_margin or r.y0 >= bottom_margin:
            continue
        # 面积太小（< 页面 2%）→ 视为 logo / 图标
        if (r.width * r.height) < page_area * 0.02:
            continue
        kept.append(r)
    kept.sort(key=lambda r: (r.y0, r.x0))
    return kept


def crop_render_png(page: fitz.Page, bbox: fitz.Rect, zoom: float = DEFAULT_ZOOM) -> bytes:
    """按坐标框裁剪渲染该区域为 PNG 字节（按看到的样子重渲染）。"""
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, clip=bbox)
    return pix.tobytes("png")


def is_blank_png(png_bytes: bytes) -> bool:
    """判断渲染产物是否「全黑/全空」（_demo 里直接抠图对象的失败特征）。

    用 PyMuPDF 解出像素，若所有样本字节相同（如全 0 全黑、全 255 全白）则视为空白。
    """
    pix = fitz.Pixmap(png_bytes)
    samples = pix.samples
    if not samples:
        return True
    first = samples[0]
    return all(b == first for b in samples)
