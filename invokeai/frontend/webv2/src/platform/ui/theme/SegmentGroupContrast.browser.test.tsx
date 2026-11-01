import { Box, ChakraProvider, SegmentGroup, Text } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { compositeColors, getContrastRatio } from './contrastRatio.testing';

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

// Segment groups sit on panels and on the subtle card surface inside them.
const SURFACES = ['bg.panel', 'bg.subtle'] as const;
const VALUES = ['media', 'assets', 'all', 'none'];

const renderSegments = async (surface: string) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box bg={surface} data-testid="surface" p="2">
          <SegmentGroup.Root value="media">
            <SegmentGroup.Indicator data-testid="indicator" />
            {VALUES.map((value) => (
              <SegmentGroup.Item key={value} data-testid={value} value={value}>
                <SegmentGroup.ItemHiddenInput />
                <SegmentGroup.ItemText display="flex" gap="1.5">
                  {value}
                  <Text as="span" color="currentColor" data-testid={`${value}-count`} opacity="0.8">
                    12
                  </Text>
                </SegmentGroup.ItemText>
              </SegmentGroup.Item>
            ))}
          </SegmentGroup.Root>
        </Box>
      </ChakraProvider>
    );
  });

  const byId = (id: string) => host!.querySelector<HTMLElement>(`[data-testid="${id}"]`)!;
  // The indicator is translucent, so measure against what it composites to over the surface beneath.
  const checkedBg = compositeColors(
    getComputedStyle(byId('indicator')).backgroundColor,
    getComputedStyle(byId('surface')).backgroundColor
  );

  return { byId, checkedBg };
};

const dividerOpacity = (item: HTMLElement) => getComputedStyle(item, '::before').opacity;

describe('segmentGroup checked contrast', () => {
  for (const theme of THEMES) {
    for (const surface of SURFACES) {
      it(`keeps checked labels readable on ${surface} in ${theme.id}`, async () => {
        applyThemeToRoot(theme.id);
        const { byId, checkedBg } = await renderSegments(surface);
        const count = getComputedStyle(byId('media-count'));

        expect(getContrastRatio(getComputedStyle(byId('media')).color, checkedBg, 1)).toBeGreaterThanOrEqual(4.5);
        // Secondary text derives from currentColor, so dimming keeps it readable on the selection.
        expect(getContrastRatio(count.color, checkedBg, Number(count.opacity))).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
});

describe('segmentGroup dividers', () => {
  it('separates unchecked neighbours and hides dividers beside the checked item', async () => {
    const { byId } = await renderSegments('bg.panel');

    // The first item has no divider and the second touches the checked item.
    expect(VALUES.map((value) => dividerOpacity(byId(value)))).toEqual(['0', '0', '1', '1']);
  });

  it('hides the dividers on both sides of a hovered item', async () => {
    const { byId } = await renderSegments('bg.panel');

    await userEvent.hover(byId('all'));

    await expect.poll(() => VALUES.map((value) => dividerOpacity(byId(value)))).toEqual(['0', '0', '0', '0']);
  });
});
