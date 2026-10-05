import { Box, ChakraProvider, Text } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { getContrastRatio } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

/**
 * Require high-contrast muted text ≥4.5:1 and boundaries ≥3:1 across themes, while keeping muted text below fg
 * emphasis.
 */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.highContrast;
  applyThemeToRoot('classic');
  host = null;
  root = null;
});

const SURFACES = ['bg.subtle', 'bg.panel'] as const;

const renderSample = async (themeId: string, highContrast: boolean) => {
  applyThemeToRoot(themeId);
  if (highContrast) {
    document.documentElement.dataset.highContrast = 'true';
  } else {
    delete document.documentElement.dataset.highContrast;
  }
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        {SURFACES.map((surface) => (
          <Box key={surface} bg={surface} data-testid={surface} p="2">
            <Text color="fg" data-testid={`${surface}-fg`}>
              fg
            </Text>
            <Text color="fg.muted" data-testid={`${surface}-muted`}>
              muted
            </Text>
            <Text color="fg.subtle" data-testid={`${surface}-subtle`}>
              subtle
            </Text>
            <Box borderColor="border" borderWidth="1px" data-testid={`${surface}-border`} />
            <Box borderColor="gray.border" borderWidth="1px" data-testid={`${surface}-gray-border`} />
          </Box>
        ))}
      </ChakraProvider>
    );
  });
  const style = (id: string) => getComputedStyle(host!.querySelector(`[data-testid="${id}"]`)!);
  return SURFACES.map((surface) => {
    const panel = style(surface).backgroundColor;
    return {
      border: getContrastRatio(style(`${surface}-border`).borderTopColor, panel, 1),
      fg: getContrastRatio(style(`${surface}-fg`).color, panel, 1),
      grayBorder: getContrastRatio(style(`${surface}-gray-border`).borderTopColor, panel, 1),
      muted: getContrastRatio(style(`${surface}-muted`).color, panel, 1),
      subtle: getContrastRatio(style(`${surface}-subtle`).color, panel, 1),
      surface,
    };
  });
};

describe('high contrast preference', () => {
  for (const theme of THEMES) {
    it(`raises muted text and borders past WCAG AA on ${theme.id}`, async () => {
      const normal = await renderSample(theme.id, false);
      await act(() => root?.unmount());
      host?.remove();
      const high = await renderSample(theme.id, true);

      high.forEach((sample, index) => {
        const before = normal[index]!;
        expect(sample.muted, `${sample.surface} muted`).toBeGreaterThan(before.muted);
        expect(sample.subtle, `${sample.surface} subtle`).toBeGreaterThan(before.subtle);
        expect(sample.border, `${sample.surface} border`).toBeGreaterThan(before.border);
        expect(sample.grayBorder, `${sample.surface} gray.border`).toBeGreaterThan(before.grayBorder);
        expect(sample.muted, `${sample.surface} muted`).toBeGreaterThanOrEqual(4.5);
        expect(sample.subtle, `${sample.surface} subtle`).toBeGreaterThanOrEqual(4.5);
        expect(sample.border, `${sample.surface} border`).toBeGreaterThanOrEqual(3);
        expect(sample.grayBorder, `${sample.surface} gray.border`).toBeGreaterThanOrEqual(3);
        // Boosted text keeps its rank: still visibly weaker than the foreground.
        expect(sample.muted, `${sample.surface} hierarchy`).toBeLessThan(sample.fg * 0.95);
      });
    });
  }
});
