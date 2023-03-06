type paletteSteps = {
  0: string;
  50: string;
  100: string;
  150: string;
  200: string;
  250: string;
  300: string;
  350: string;
  400: string;
  450: string;
  500: string;
  550: string;
  600: string;
  650: string;
  700: string;
  750: string;
  800: string;
  850: string;
  900: string;
  950: string;
  1000: string;
};

/**
 * Add two numbers together
 * @param  {String | Number} hue Hue of the color (0-360) - Reds 0, Greens 120, Blues 240
 * @param  {String | Number} saturation Saturation of the color (0-100)
 * @param  {boolean} light True to generate light color palette
 */
export function generateColorPalette(
  hue: string | number,
  saturation: string | number,
  light = false
) {
  hue = String(hue);
  saturation = String(saturation);

  const colorSteps = Array.from({ length: 21 }, (_, i) => i * 50);
  const lightnessSteps = [
    '0',
    '5',
    '10',
    '15',
    '20',
    '25',
    '30',
    '35',
    '40',
    '45',
    '50',
    '55',
    '59',
    '64',
    '68',
    '73',
    '77',
    '82',
    '86',
    '95',
    '100',
  ];

  const darkPalette: Partial<paletteSteps> = {};
  const lightPalette: Partial<paletteSteps> = {};

  colorSteps.forEach((colorStep, index) => {
    darkPalette[
      colorStep as keyof typeof darkPalette
    ] = `hsl(${hue}, ${saturation}%, ${
      lightnessSteps[colorSteps.length - 1 - index]
    }%)`;

    lightPalette[
      colorStep as keyof typeof lightPalette
    ] = `hsl(${hue}, ${saturation}%, ${lightnessSteps[index]}%)`;
  });

  return light ? lightPalette : darkPalette;
}
