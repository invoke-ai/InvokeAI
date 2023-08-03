import { InvokeAIPaletteSteps } from 'theme/themeTypes';

/**
 * Add two numbers together
 * @param  {String | Number} H Hue of the color (0-360) - Reds 0, Greens 120, Blues 240
 * @param  {String | Number} L Saturation of the color (0-100)
 * @param  {Boolean} alpha Whether or not to generate this palette as a transparency palette
 */
export function generateColorPalette(
  H: string | number,
  S: string | number,
  alpha = false
) {
  H = String(H);
  S = String(S);

  const colorSteps = Array.from({ length: 21 }, (_, i) => i * 50);

  const lightnessSteps = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 59, 64, 68, 73, 77, 82, 86,
    95, 100,
  ];

  const p = colorSteps.reduce((palette, step, index) => {
    const A = alpha ? (lightnessSteps[index] as number) / 100 : 1;

    // Lightness should be 50% for alpha colors
    const L = alpha ? 50 : lightnessSteps[colorSteps.length - 1 - index];

    palette[step as keyof typeof palette] = `hsl(${H} ${S}% ${L}% / ${A})`;

    return palette;
  }, {} as InvokeAIPaletteSteps);

  return p;
}
