import { ChakraProvider, SegmentGroup, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { getContrastRatio } from './contrastRatio.testing';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderSegments = async (): Promise<{ checked: HTMLElement; count: HTMLElement; indicator: HTMLElement }> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <SegmentGroup.Root size="xs" value="media">
          <SegmentGroup.Indicator data-testid="indicator" />
          {['media', 'assets'].map((value) => (
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
      </ChakraProvider>
    );
  });

  return {
    checked: host.querySelector<HTMLElement>('[data-testid="media"]')!,
    count: host.querySelector<HTMLElement>('[data-testid="media-count"]')!,
    indicator: host.querySelector<HTMLElement>('[data-testid="indicator"]')!,
  };
};

describe('segmentGroup checked contrast', () => {
  // The indicator is a solid accent fill, so anything inside the checked item
  // has to be readable against accent — not against the panel behind it.
  it('renders the checked label against the accent fill at AA contrast', async () => {
    const { checked, indicator } = await renderSegments();

    const ratio = getContrastRatio(getComputedStyle(checked).color, getComputedStyle(indicator).backgroundColor, 1);

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps text that inherits the item colour readable when dimmed', async () => {
    const { count, indicator } = await renderSegments();
    const style = getComputedStyle(count);

    // Secondary text inside a segment must dim from `currentColor`; pinning it
    // to a fixed `fg.muted` grey drops to ~1.5:1 on the accent fill.
    const ratio = getContrastRatio(style.color, getComputedStyle(indicator).backgroundColor, Number(style.opacity));

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});
