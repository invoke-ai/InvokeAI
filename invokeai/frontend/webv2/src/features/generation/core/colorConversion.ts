// SPDX-License-Identifier: Apache-2.0
export type RgbImage = {
  readonly width: number;
  readonly height: number;
  /** Interleaved byte-valued red, green, blue samples in row-major pixel order. */
  readonly data: Uint8Array;
};

export type OklabImage = {
  readonly width: number;
  readonly height: number;
  /** Interleaved lightness, a, b samples in row-major pixel order. */
  readonly data: Float64Array;
};

const decodeSrgb = (channel: number): number => {
  const normalized = channel / 255;
  return normalized <= 0.04045 ? normalized / 12.92 : ((normalized + 0.055) / 1.055) ** 2.4;
};

/** Converts interleaved byte-valued sRGB pixels to compact interleaved Oklab samples. */
export const oklabFromSrgb = (rgb: RgbImage): OklabImage => {
  const { width, height, data: input } = rgb;
  if (
    !Number.isInteger(width) ||
    width < 1 ||
    !Number.isInteger(height) ||
    height < 1 ||
    !(input instanceof Uint8Array) ||
    input.length !== width * height * 3
  ) {
    throw new Error('rgb must be a non-empty image with three interleaved byte channels per pixel');
  }

  const data = new Float64Array(input.length);
  for (let i = 0; i < input.length; i += 3) {
    const r = decodeSrgb(input[i]);
    const g = decodeSrgb(input[i + 1]);
    const b = decodeSrgb(input[i + 2]);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    data[i] = 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
    data[i + 1] = 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s;
    data[i + 2] = 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s;
  }
  return { width, height, data };
};
