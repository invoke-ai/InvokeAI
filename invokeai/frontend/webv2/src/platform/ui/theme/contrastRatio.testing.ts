/**
 * WCAG contrast measurement for accessibility assertions in browser tests.
 *
 * Lives here rather than in each test because it is a fixed formula, not a
 * per-surface judgement: two copies of it can silently disagree about what a
 * ratio *is*, and a contrast test that measures wrong is worse than no test —
 * it reports the palette as accessible when it is not.
 *
 * Test-only. It resolves colours by painting them to a canvas, so it needs a
 * real browser and has no place in the app bundle.
 */

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

/**
 * Contrast of `foreground` at `alpha` composited over `background`.
 *
 * Opacity is applied by compositing rather than by scaling the ratio, because
 * a translucent foreground is read against the pixel it actually produces.
 */
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
