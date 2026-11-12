import { Box, Button, ChakraProvider, Text } from '@chakra-ui/react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { server } from 'vitest/browser';

import { applyThemeToRoot } from './applyTheme';
import { compositeColors, getContrastRatio, toRgb } from './contrastRatio.testing';
import { system } from './system';

declare module 'vitest/browser' {
  interface BrowserCommands {
    setBrowserColorScheme: (scheme: 'light' | 'dark' | null) => Promise<void>;
  }
}

let host: HTMLDivElement | undefined;
let root: Root | undefined;
let desktopStyle: HTMLStyleElement | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950] as const;
const PALETTES = ['brand', 'accent', 'red', 'green', 'orange', 'blue'] as const;

const renderSample = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <Box bg="bg.subtle" data-testid="surface" p="4">
          <Text color="fg" data-testid="fg">
            Foreground
          </Text>
          <Text color="fg.muted" data-testid="muted">
            Muted
          </Text>
          <Text color="fg.subtle" data-testid="subtle">
            Subtle
          </Text>
          <Box bg="bg.inset" data-testid="inset" />
          <Box bg="bg.emphasized" data-testid="control" />
          <Box bg="bg.panel" data-testid="panel" />
          <Box bg="bg.muted" data-testid="muted-panel" />
          <Box bg="gray.subtle" data-testid="fill" />
          <Box borderColor="border" borderWidth="1px" data-testid="border" />
          {STEPS.map((step) => (
            <Box key={step} bg={`neutral.${step}`} data-testid={`neutral-${step}`} />
          ))}
          {PALETTES.map((palette) => (
            <Box key={palette}>
              <Button colorPalette={palette} data-testid={palette} transition="none">
                Action
              </Button>
              <Text color={`${palette}.fg`} data-testid={`${palette}-fg`}>
                Status
              </Text>
            </Box>
          ))}
          <Box bg="Canvas" color="CanvasText" data-testid="native" />
          <Box bg="LinkText" color="Canvas" data-testid="native-link" />
        </Box>
      </ChakraProvider>
    )
  );
};

const style = (id: string) => getComputedStyle(host!.querySelector(`[data-testid="${id}"]`)!);
const paletteStyles = () =>
  Object.fromEntries(
    PALETTES.map((palette) => [
      palette,
      {
        color: style(palette).color,
        foreground: style(`${palette}-fg`).color,
        background: style(palette).backgroundColor,
      },
    ])
  );

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  desktopStyle?.remove();
  desktopStyle = undefined;
  root = undefined;
  host = undefined;
  delete document.documentElement.dataset.highContrast;
  applyThemeToRoot('classic');
  await server.commands.setBrowserColorScheme(null);
});

describe('System theme', () => {
  for (const scheme of ['light', 'dark'] as const) {
    it(`resolves browser colors and readable controls in ${scheme}`, async () => {
      await server.commands.setBrowserColorScheme(scheme);
      applyThemeToRoot('system');
      await renderSample();
      expect(toRgb(style('fg').color)).toEqual(toRgb(style('native').color));
      expect(toRgb(style('inset').backgroundColor)).toEqual(toRgb(style('native').backgroundColor));
      expect(toRgb(style('brand').backgroundColor)).toEqual(toRgb(style('native-link').backgroundColor));
      expect(toRgb(style('brand').color)).toEqual(toRgb(style('native-link').color));
      const surface = style('surface').backgroundColor;
      expect(getContrastRatio(style('fg').color, surface, 1)).toBeGreaterThanOrEqual(4.5);
      expect(getContrastRatio(style('muted').color, surface, 1)).toBeGreaterThanOrEqual(4.5);
      expect(getContrastRatio(style('control').backgroundColor, surface, 1)).toBeGreaterThan(1.05);
      expect(getContrastRatio(style('brand-fg').color, style('control').backgroundColor, 1)).toBeGreaterThanOrEqual(3);
      for (const palette of PALETTES) {
        expect(
          getContrastRatio(style(palette).color, compositeColors(style(palette).backgroundColor, surface), 1),
          palette
        ).toBeGreaterThanOrEqual(4.5);
      }
      const luminances = STEPS.map((step) =>
        toRgb(style(`neutral-${step}`).backgroundColor).reduce(
          (sum, channel, index) => sum + channel * [0.2126, 0.7152, 0.0722][index]!,
          0
        )
      );
      for (let index = 1; index < luminances.length; index++) {
        expect(luminances[index], `neutral.${STEPS[index]}`).toBeLessThan(luminances[index - 1]!);
      }
      document.documentElement.dataset.highContrast = 'true';
      expect(getContrastRatio(style('subtle').color, surface, 1)).toBeGreaterThanOrEqual(4.5);
      expect(getContrastRatio(style('border').borderTopColor, surface, 1)).toBeGreaterThanOrEqual(3);
    });
  }

  for (const palette of [
    {
      mode: 'dark',
      background: '#2c2525',
      foreground: '#e6d9db',
      accent: '#f38d70',
      contrast: '#2c2525',
      surface: '#3d2f2a',
      inset: '#211b1b',
      selection: '#403e41',
    },
    {
      mode: 'light',
      background: '#faf4ed',
      foreground: '#575279',
      accent: '#56949f',
      contrast: '#000000',
      surface: '#f2e9e1',
      inset: '#ede7e1',
      selection: '#dfdad9',
    },
    {
      mode: 'dark',
      background: '#222222',
      foreground: '#eeeeee',
      accent: '#777777',
      contrast: '#000000',
      surface: '#2e2e2e',
      inset: '#222222',
      selection: '#363636',
    },
    {
      mode: 'light',
      background: '#ffffff',
      foreground: '#222222',
      accent: '#006b5f',
      contrast: '#ffffff',
      surface: '#f2f2f2',
      inset: '#ffffff',
      selection: '#e9e9e9',
    },
  ] as const) {
    it(`uses the supplied ${palette.mode} desktop palette ${palette.accent} even against the browser preference`, async () => {
      await server.commands.setBrowserColorScheme(palette.mode === 'dark' ? 'light' : 'dark');
      desktopStyle = document.createElement('style');
      desktopStyle.textContent = `:root {
        --invokeai-desktop-background: ${palette.background};
        --invokeai-desktop-foreground: ${palette.foreground};
        --invokeai-desktop-accent: ${palette.accent};
        --invokeai-desktop-accent-contrast: ${palette.contrast};
        --invokeai-desktop-color-scheme: ${palette.mode};
        --invokeai-desktop-surface: ${palette.surface};
        --invokeai-desktop-inset: ${palette.inset};
        --invokeai-desktop-selection: ${palette.selection};
      }`;
      document.head.append(desktopStyle);
      applyThemeToRoot('system');
      await renderSample();
      expect(getComputedStyle(document.documentElement).colorScheme).toBe(palette.mode);
      expect(toRgb(style('inset').backgroundColor)).toEqual(toRgb(palette.inset));
      expect(toRgb(style('surface').backgroundColor)).toEqual(toRgb(palette.background));
      expect(toRgb(style('panel').backgroundColor)).toEqual(toRgb(palette.surface));
      expect(toRgb(style('muted-panel').backgroundColor)).toEqual(toRgb(palette.surface));
      expect(toRgb(style('control').backgroundColor)).toEqual(toRgb(palette.surface));
      expect(toRgb(style('fill').backgroundColor)).toEqual(toRgb(palette.selection));
      expect(toRgb(style('fg').color)).toEqual(toRgb(palette.foreground));
      expect(toRgb(style('brand-fg').color)).toEqual(toRgb(palette.foreground));
      expect(toRgb(style('brand').backgroundColor)).toEqual(toRgb(palette.accent));
      expect(toRgb(style('brand').color)).toEqual(toRgb(palette.contrast));
      for (const action of PALETTES) {
        expect(
          getContrastRatio(style(action).color, compositeColors(style(action).backgroundColor, palette.background), 1),
          `${action} action on ${palette.accent}`
        ).toBeGreaterThanOrEqual(4.5);
      }
      expect(getContrastRatio(style('fg').color, style('surface').backgroundColor, 1)).toBeGreaterThanOrEqual(4.5);
      for (const surface of ['surface', 'panel']) {
        expect(getContrastRatio(style('muted').color, style(surface).backgroundColor, 1)).toBeGreaterThanOrEqual(4.5);
        expect(
          getContrastRatio(style('subtle').color, style(surface).backgroundColor, 1),
          `subtle text on ${surface}`
        ).toBeGreaterThanOrEqual(4.5);
      }
      document.documentElement.dataset.highContrast = 'true';
      for (const surface of ['surface', 'panel']) {
        expect(getContrastRatio(style('muted').color, style(surface).backgroundColor, 1)).toBeGreaterThanOrEqual(4.5);
        expect(getContrastRatio(style('subtle').color, style(surface).backgroundColor, 1)).toBeGreaterThanOrEqual(4.5);
        expect(
          getContrastRatio(style('border').borderTopColor, style(surface).backgroundColor, 1)
        ).toBeGreaterThanOrEqual(3);
      }
      delete document.documentElement.dataset.highContrast;
      applyThemeToRoot('classic');
      const fixedColors = paletteStyles();
      const fixedBackground = style('inset').backgroundColor;
      desktopStyle.remove();
      expect(paletteStyles()).toEqual(fixedColors);
      expect(style('inset').backgroundColor).toBe(fixedBackground);
    });
  }

  it('follows preference changes live while fixed themes stay fixed', async () => {
    await server.commands.setBrowserColorScheme('light');
    applyThemeToRoot('system');
    await renderSample();
    const lightForeground = style('fg').color;
    const lightPalettes = paletteStyles();
    await server.commands.setBrowserColorScheme('dark');
    await expect.poll(() => style('fg').color).not.toBe(lightForeground);
    expect(document.documentElement.dataset.theme).toBe('system');
    expect(paletteStyles().red).not.toEqual(lightPalettes.red);
    expect(getContrastRatio(style('fg').color, style('surface').backgroundColor, 1)).toBeGreaterThanOrEqual(4.5);
    applyThemeToRoot('classic');
    const fixedForeground = style('fg').color;
    const fixedPalettes = paletteStyles();
    await server.commands.setBrowserColorScheme('light');
    expect(style('fg').color).toBe(fixedForeground);
    expect(paletteStyles()).toEqual(fixedPalettes);
  });
});
