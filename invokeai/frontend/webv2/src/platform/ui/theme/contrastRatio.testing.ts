/** Browser-only contrast helpers; resolve colors through canvas and keep out of production bundles. */

/** Resolve any CSS colour — named, `oklch()`, `color-mix()` — to sRGB channels. */
export const toRgb = (color: string): [number, number, number] => {
  const context = document.createElement('canvas').getContext('2d')!;

  context.fillStyle = color;
  context.fillRect(0, 0, 1, 1);

  const [red, green, blue] = context.getImageData(0, 0, 1, 1).data;

  return [red!, green!, blue!];
};

const getRelativeLuminance = ([red, green, blue]: [number, number, number]): number => {
  const linearize = (channel: number): number => {
    const value = channel / 255;

    return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
  };

  return 0.2126 * linearize(red) + 0.7152 * linearize(green) + 0.0722 * linearize(blue);
};

/** Composite alpha before measuring contrast; never scale the ratio itself. */
export const getContrastRatio = (foreground: string, background: string, alpha: number): number => {
  const backgroundRgb = toRgb(background);
  const composited = toRgb(foreground).map((channel, index) =>
    Math.round(channel * alpha + backgroundRgb[index]! * (1 - alpha))
  ) as [number, number, number];
  const [lighter, darker] = [getRelativeLuminance(composited), getRelativeLuminance(backgroundRgb)].sort(
    (a, b) => b - a
  );

  return (lighter! + 0.05) / (darker! + 0.05);
};

/** Paint translucent fills over the surface; raw channel parsing would ignore compositing. */
export const compositeColors = (top: string, bottom: string): string => {
  const context = document.createElement('canvas').getContext('2d')!;

  context.fillStyle = bottom;
  context.fillRect(0, 0, 1, 1);
  context.fillStyle = top;
  context.fillRect(0, 0, 1, 1);

  const [red, green, blue] = context.getImageData(0, 0, 1, 1).data;

  return `rgb(${String(red)}, ${String(green)}, ${String(blue)})`;
};
