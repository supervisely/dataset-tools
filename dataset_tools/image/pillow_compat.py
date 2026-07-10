from PIL import ImageDraw, ImageFont


def _bbox_size(bbox):
    left, top, right, bottom = bbox
    return right - left, bottom - top


def font_text_size(font: ImageFont.FreeTypeFont, text: str):
    if hasattr(font, "getbbox"):
        return _bbox_size(font.getbbox(text))
    return font.getsize(text)


def draw_text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont):
    if hasattr(draw, "textbbox"):
        return _bbox_size(draw.textbbox((0, 0), text, font=font))
    return draw.textsize(text, font=font)
