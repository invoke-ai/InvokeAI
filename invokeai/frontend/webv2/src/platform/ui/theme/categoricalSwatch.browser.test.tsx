import { Box, ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { getContrastRatio } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

const PALETTES = [
  'aqua',
  'azure',
  'coral',
  'gold',
  'ice',
  'lavender',
  'lime',
  'mint',
  'periwinkle',
  'sage',
  'silver',
  'steel',
];
const SURFACES = ['bg', 'bg.subtle', 'bg.panel'] as const;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  applyThemeToRoot('classic');
  host = null;
  root = null;
});

describe('categorical swatches', () => {
  for (const theme of THEMES) {
    it(`stand out from every surface on ${theme.id}, by their fill or their fg edge`, async () => {
      applyThemeToRoot(theme.id);
      host = document.createElement('div');
      document.body.append(host);
      root = createRoot(host);
      await act(() => {
        root?.render(
          <ChakraProvider value={system}>
            {SURFACES.map((surface) => (
              <Box key={surface} bg={surface} data-surface={surface}>
                {PALETTES.map((palette) => (
                  <Box key={palette} bg={`${palette}.swatch`} borderColor={`${palette}.fg`} data-palette={palette} />
                ))}
              </Box>
            ))}
          </ChakraProvider>
        );
      });

      for (const surface of SURFACES) {
        const surfaceNode = host.querySelector(`[data-surface="${surface}"]`)!;
        const background = getComputedStyle(surfaceNode).backgroundColor;

        for (const palette of PALETTES) {
          const swatch = getComputedStyle(surfaceNode.querySelector(`[data-palette="${palette}"]`)!);
          const ratio = Math.max(
            getContrastRatio(swatch.backgroundColor, background, 1),
            getContrastRatio(swatch.borderTopColor, background, 1)
          );

          expect(ratio, `${palette} on ${surface}`).toBeGreaterThanOrEqual(3);
        }
      }
    });
  }
});
