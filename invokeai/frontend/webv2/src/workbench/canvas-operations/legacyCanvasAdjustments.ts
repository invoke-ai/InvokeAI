import { z } from 'zod';

const unit = z.number().finite().min(-1).max(1);
const simple = z.object({
  brightness: unit,
  contrast: unit,
  saturation: unit,
  temperature: unit,
  tint: unit,
  sharpness: unit,
});
const curve = z
  .array(z.tuple([z.number().min(0).max(255), z.number().min(0).max(255)]))
  .min(2)
  .max(256);
const curves = z.object({ master: curve, r: curve, g: curve, b: curve });
const adjustments = z.object({
  version: z.literal(1),
  enabled: z.boolean(),
  mode: z.enum(['simple', 'curves']),
  simple,
  curves,
});
const clamp = (value: number): number => Math.max(0, Math.min(255, value));

const curveLut = (points: [number, number][]): number[] => {
  const sorted = points.map(([x, y]): [number, number] => [Math.round(x), Math.round(y)]).sort((a, b) => a[0] - b[0]);
  if (sorted[0]![0] !== 0) {
    sorted.unshift([0, sorted[0]![1]]);
  }
  if (sorted.at(-1)![0] !== 255) {
    sorted.push([255, sorted.at(-1)![1]]);
  }
  let segment = 0;
  return Array.from({ length: 256 }, (_, x) => {
    while (segment < sorted.length - 2 && x > sorted[segment + 1]![0]) {
      segment++;
    }
    const [x0, y0] = sorted[segment]!,
      [x1, y1] = sorted[segment + 1]!;
    return clamp(Math.round(y0 + (y1 - y0) * (x1 === x0 ? 0 : (x - x0) / (x1 - x0))));
  });
};

/** Match the legacy committed-layer filter, baking it once into its recalled paint bitmap. */
export const applyLegacyCanvasAdjustments = (pixels: ImageData, raw: unknown): void => {
  if (raw === undefined) {
    return;
  }
  const config = adjustments.parse(raw);
  if (!config.enabled) {
    return;
  }
  const { data, width, height } = pixels;
  if (config.mode === 'curves') {
    const luts = {
      master: curveLut(config.curves.master),
      r: curveLut(config.curves.r),
      g: curveLut(config.curves.g),
      b: curveLut(config.curves.b),
    };
    for (let i = 0; i < data.length; i += 4) {
      data[i] = luts.r[luts.master[data[i]!]!]!;
      data[i + 1] = luts.g[luts.master[data[i + 1]!]!]!;
      data[i + 2] = luts.b[luts.master[data[i + 2]!]!]!;
    }
    return;
  }
  const { brightness, contrast, saturation, temperature, tint, sharpness } = config.simple;
  const tintScale = tint * 0.5,
    sat = 1 + saturation;
  for (let i = 0; i < data.length; i += 4) {
    const r = ((data[i]! + brightness * 255 - 128) * (1 + contrast) + 128) * (1 + temperature * 0.5) * (1 + tintScale);
    const g = ((data[i + 1]! + brightness * 255 - 128) * (1 + contrast) + 128) * (1 - tintScale);
    const b =
      ((data[i + 2]! + brightness * 255 - 128) * (1 + contrast) + 128) * (1 - temperature * 0.5) * (1 + tintScale);
    const luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
    data[i] = clamp(luma * (1 - sat) + r * sat);
    data[i + 1] = clamp(luma * (1 - sat) + g * sat);
    data[i + 2] = clamp(luma * (1 - sat) + b * sat);
  }
  if (Math.abs(sharpness) <= 0.001 || width <= 2 || height <= 2) {
    return;
  }
  const source = new Uint8ClampedArray(data),
    amount = sharpness * 0.5;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        data[index + channel] = clamp(
          source[index + channel]! * (1 + 4 * amount) -
            amount *
              (source[index - 4 + channel]! +
                source[index + 4 + channel]! +
                source[index - width * 4 + channel]! +
                source[index + width * 4 + channel]!)
        );
      }
    }
  }
};
