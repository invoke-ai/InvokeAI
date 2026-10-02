import { Badge, Box, ChakraProvider, HStack } from '@chakra-ui/react';
import { MODEL_BASES } from '@features/models/core/baseIdentity';
import { applyThemeToRoot } from '@theme/applyTheme';
import { getContrastRatio, toRgb } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';

import { ModelBaseBadge } from './ModelBadges';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const UPDATED_BASES = new Set([
  'sd-2',
  'sdxl-refiner',
  'flux2',
  'z-image',
  'ideogram-4',
  'krea-2',
  'wan',
  'minimax-h3',
  'ltx-2',
]);
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  applyThemeToRoot('classic');
});

it('keeps base badges distinct and readable through theme changes and filter selection', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box bg="bg.panel" p="4" width="400px">
          {Object.values(MODEL_BASES).map(({ base, label, colorPalette }) => (
            <HStack key={base} data-base={base} gap="4" mb="2">
              <ModelBaseBadge base={base} />
              <Badge colorPalette={colorPalette} size="sm" variant="solid">
                {label}
              </Badge>
            </HStack>
          ))}
        </Box>
      </ChakraProvider>
    );
  });

  for (const theme of THEMES) {
    applyThemeToRoot(theme.id);
    const identities = new Set<string>();
    for (const { base, colorPalette } of Object.values(MODEL_BASES)) {
      const badges = host.querySelectorAll(`[data-base="${base}"] .chakra-badge`);
      expect(badges).toHaveLength(2);
      const surface = getComputedStyle(badges[0]!);
      const solid = getComputedStyle(badges[1]!);
      if (colorPalette !== 'gray') {
        const identity = toRgb(surface.color).join(',');
        expect(identities.has(identity), `${theme.id}: duplicate ${base} foreground`).toBe(false);
        identities.add(identity);
      }
      if (UPDATED_BASES.has(base)) {
        for (const [variant, style] of [
          ['surface', surface],
          ['solid', solid],
        ] as const) {
          expect(
            getContrastRatio(style.color, style.backgroundColor, 1),
            `${theme.id}: ${base} ${variant} text contrast`
          ).toBeGreaterThanOrEqual(4.5);
        }
      }
      if (base === 'ideogram-4' && theme.colorScheme === 'dark') {
        expect(toRgb(surface.color)).toEqual([189, 159, 255]);
      }
    }
  }
});
