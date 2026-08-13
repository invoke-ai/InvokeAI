import { ChakraProvider, SegmentGroup, Text } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const toRgb = (color: string): [number, number, number] => {
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

const getContrastRatio = (foreground: string, background: string, alpha: number): number => {
  const backgroundRgb = toRgb(background);
  const foregroundRgb = toRgb(foreground);
  // Opacity is applied by compositing, so measure the blended pixel.
  const composited = foregroundRgb.map((channel, index) =>
    Math.round(channel * alpha + backgroundRgb[index]! * (1 - alpha))
  ) as [number, number, number];
  const [lighter, darker] = [getRelativeLuminance(composited), getRelativeLuminance(backgroundRgb)].sort(
    (a, b) => b - a
  );

  return (lighter! + 0.05) / (darker! + 0.05);
};

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
