import { Box, ChakraProvider } from '@chakra-ui/react';
import { Row } from '@platform/ui';
import { applyThemeToRoot } from '@theme/applyTheme';
import { getContrastRatio } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { WIDGET_ITEM_SX } from './WidgetBar';

/**
 * The rail's active state is a neutral fill with the widget icon in
 * `brand.fg`. Two things can silently break it, so both are pinned here:
 * `brand.fg` reverting to the raw seed (1.20:1 on the light theme's panel —
 * invisible), and the active fill being swapped for a brand tint, which on the
 * light theme lands within 1.06:1 of the rail behind it.
 */

const RAIL_ITEM_SX = WIDGET_ITEM_SX;

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

const renderActiveRailItem = async (themeId: string) => {
  applyThemeToRoot(themeId);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box bg="bg.subtle" data-testid="rail" p="1">
          <Row aria-pressed css={RAIL_ITEM_SX} data-testid="item" />
        </Box>
      </ChakraProvider>
    );
  });

  return {
    item: host.querySelector<HTMLElement>('[data-testid="item"]')!,
    rail: host.querySelector<HTMLElement>('[data-testid="rail"]')!,
  };
};

describe('rail active state contrast', () => {
  for (const theme of THEMES) {
    // 3:1 is the WCAG floor for a graphical control's own colours; the icon is
    // an icon, not body text.
    it(`keeps the active icon readable on ${theme.id}`, async () => {
      const { item } = await renderActiveRailItem(theme.id);
      const style = getComputedStyle(item);

      expect(getContrastRatio(style.color, style.backgroundColor, 1)).toBeGreaterThanOrEqual(3);
    });

    it(`separates the active fill from the rail behind it on ${theme.id}`, async () => {
      const { item, rail } = await renderActiveRailItem(theme.id);

      expect(
        getContrastRatio(getComputedStyle(item).backgroundColor, getComputedStyle(rail).backgroundColor, 1)
      ).toBeGreaterThan(1.05);
    });
  }
});
