# Copyright (c) 2021 W-Mai
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# lvgl_image_converter/lv_img_converter/lv_img_converter.py
# Created by W-Mai on 2021/2/17.
# repo: https://github.com/W-Mai/lvgl_image_converter
#
##############################################################
import struct
from typing import Optional, List, Any, Dict, NoReturn, AnyStr
from decimal import Decimal, ROUND_HALF_DOWN
from dataclasses import dataclass
from enum import IntEnum
from PIL import Image
import numpy as np

class ColorFormat(IntEnum):
    """Enumeration of supported color formats"""
    TRUE_COLOR_332 = 0
    TRUE_COLOR_565 = 1
    TRUE_COLOR_565_SWAP = 2
    TRUE_COLOR_888 = 3
    ALPHA_1_BIT = 4
    ALPHA_2_BIT = 5
    ALPHA_4_BIT = 6
    ALPHA_8_BIT = 7
    INDEXED_1_BIT = 8
    INDEXED_2_BIT = 9
    INDEXED_4_BIT = 10
    INDEXED_8_BIT = 11
    RAW = 12
    RAW_ALPHA = 13
    RAW_CHROMA = 14
    TRUE_COLOR = 100
    TRUE_COLOR_ALPHA = 101
    TRUE_COLOR_CHROMA = 102

@dataclass
class ImageData:
    """Container for image processing data"""
    width: int
    height: int
    data: List[int]
    palette: Optional[List[int]] = None

class LVGLConverter:
    def __init__(
        self,
        path: str,
        out_name: str,
        dither: bool = True,
        cf: ColorFormat = ColorFormat.INDEXED_4_BIT,
        cf_palette_bgr_en: bool = True
    ):
        self.path = path
        self.out_name = out_name
        self.cf = cf
        self.dither = dither
        self.cf_palette_bgr_en = cf_palette_bgr_en

        # Initialize image data
        self.img: Optional[Image.Image] = None
        self.width: int = 0
        self.height: int = 0
        self.output_data: List[int] = []

        # Dithering buffers
        self.r_err_next: float = 0
        self.g_err_next: float = 0
        self.b_err_next: float = 0
        self.error_diffusion: Optional[np.ndarray] = None

        self._init_image()

    def _init_image(self) -> None:
        """Initialize image and related data"""
        if self.cf in {ColorFormat.RAW, ColorFormat.RAW_ALPHA, ColorFormat.RAW_CHROMA}:
            return

        self.img = Image.open(self.path)
        self.width, self.height = self.img.size

        if self.dither:
            # Pre-allocate error diffusion buffers using numpy for better performance
            self.error_diffusion = np.zeros((3, self.width + 2), dtype=np.float32)

    @staticmethod
    def _get_color_from_palette(palette: List[int], index: int) -> List[int]:
        """Extract RGB values from palette"""
        start = 3 * index
        return [palette[start + i] for i in range(3)]

    def _classify_pixel(self, value: float, bits: int) -> int:
        """Quantize pixel value to specified bit depth"""
        tmp = 1 << (8 - bits)
        val = int(Decimal(str(value / tmp)).quantize(Decimal('0'), rounding=ROUND_HALF_DOWN)) * tmp
        return max(0, val)

    def _process_pixel(self, x: int, y: int) -> None:
        """Process a single pixel"""
        pixel = self.img.getpixel((x, y))

        # Handle palette images
        if self.img.mode == "P":
            pixel = self._get_color_from_palette(self.img.getpalette(), pixel)

        # Extract color components
        r, g, b = pixel[:3]
        alpha = pixel[3] if len(pixel) == 4 and self.dither else 0xFF

        # Apply dithering if enabled
        if self.dither:
            r, g, b = self._apply_dithering(r, g, b, x)

        self._write_pixel(x, y, r, g, b, alpha)

    def _apply_dithering(self, r: int, g: int, b: int, x: int) -> tuple[int, int, int]:
        """Apply error diffusion dithering"""
        if not self.dither:
            return r, g, b

        # Add accumulated errors
        r = r + self.r_err_next + self.error_diffusion[0, x + 1]
        g = g + self.g_err_next + self.error_diffusion[1, x + 1]
        b = b + self.b_err_next + self.error_diffusion[2, x + 1]

        # Clear used error
        self.error_diffusion[:, x + 1] = 0

        # Quantize based on color format
        if self.cf == ColorFormat.TRUE_COLOR_332:
            r = min(0xE0, self._classify_pixel(r, 3))
            g = min(0xE0, self._classify_pixel(g, 3))
            b = min(0xC0, self._classify_pixel(b, 2))
        elif self.cf in {ColorFormat.TRUE_COLOR_565, ColorFormat.TRUE_COLOR_565_SWAP}:
            r = min(0xF8, self._classify_pixel(r, 5))
            g = min(0xFC, self._classify_pixel(g, 6))
            b = min(0xF8, self._classify_pixel(b, 5))
        else:  # TRUE_COLOR_888
            r = min(0xFF, self._classify_pixel(r, 8))
            g = min(0xFF, self._classify_pixel(g, 8))
            b = min(0xFF, self._classify_pixel(b, 8))

        # Calculate and distribute errors
        r_err = r - r
        g_err = g - g
        b_err = b - b

        # Store errors for next pixel
        self.r_err_next = (7 * r_err) / 16
        self.g_err_next = (7 * g_err) / 16
        self.b_err_next = (7 * b_err) / 16

        # Distribute errors to adjacent pixels
        for i, err in enumerate([r_err, g_err, b_err]):
            self.error_diffusion[i, x] += (3 * err) / 16
            self.error_diffusion[i, x + 1] += (5 * err) / 16
            self.error_diffusion[i, x + 2] += err / 16

        return r, g, b

    def convert(self, cf: Optional[ColorFormat] = None, alpha: int = 0) -> None:
        """Convert image to specified format"""
        if cf is not None:
            self.cf = cf

        # Handle raw formats
        if self.cf in {ColorFormat.RAW, ColorFormat.RAW_ALPHA, ColorFormat.RAW_CHROMA}:
            with open(self.path, "rb") as f:
                self.output_data = list(f.read())
            return

        # Process each pixel
        for y in range(self.height):
            self._reset_dithering()
            for x in range(self.width):
                self._process_pixel(x, y)

    def _reset_dithering(self) -> None:
        """Reset dithering errors for new row"""
        if self.dither:
            self.r_err_next = 0
            self.g_err_next = 0
            self.b_err_next = 0

    def get_bin_file(self, cf: Optional[ColorFormat] = None) -> bytes:
        """Generate binary output file"""
        if cf is None:
            cf = self.cf

        # Map color format to LVGL format
        lv_cf_map = {
            ColorFormat.TRUE_COLOR: 4,
            ColorFormat.TRUE_COLOR_ALPHA: 5,
            ColorFormat.TRUE_COLOR_CHROMA: 6,
            ColorFormat.INDEXED_1_BIT: 7,
            ColorFormat.INDEXED_2_BIT: 8,
            ColorFormat.INDEXED_4_BIT: 9,
            ColorFormat.INDEXED_8_BIT: 10,
            ColorFormat.ALPHA_1_BIT: 11,
            ColorFormat.ALPHA_2_BIT: 12,
            ColorFormat.ALPHA_4_BIT: 13,
            ColorFormat.ALPHA_8_BIT: 14,
        }

        lv_cf = lv_cf_map.get(cf, 4)
        header = lv_cf + (self.width << 10) + (self.height << 21)

        return struct.pack("<L", header) + struct.pack(f"<{len(self.output_data)}B", *self.output_data)

export default LVGLConverter
