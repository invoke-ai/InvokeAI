/**
 * Konva filters
 * https://konvajs.org/docs/filters/Custom_Filter.html
 */

/**
 * Calculates the lightness (HSL) of a given pixel and sets the alpha channel to that value.
 * This is useful for edge maps and other masks, to make the black areas transparent.
 * @param imageData The image data to apply the filter to
 */
export const LightnessToAlphaFilter = (imageData: ImageData): void => {
  const len = imageData.data.length / 4;
  for (let i = 0; i < len; i++) {
    const r = imageData.data[i * 4 + 0] as number;
    const g = imageData.data[i * 4 + 1] as number;
    const b = imageData.data[i * 4 + 2] as number;
    const a = imageData.data[i * 4 + 3] as number;
    const cMin = Math.min(r, g, b);
    const cMax = Math.max(r, g, b);
    imageData.data[i * 4 + 3] = Math.min(a, (cMin + cMax) / 2);
  }
};
