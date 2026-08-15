import { Button, ChakraProvider, SegmentGroup } from '@chakra-ui/react';
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

const renderPair = async (
  size: '2xs' | 'xs' | 'sm' | 'md'
): Promise<{ button: HTMLElement; segmentRoot: HTMLElement }> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Button data-testid="button" size={size}>
          Reference
        </Button>
        {/* `2xs` is a repo recipe extension the generated Chakra types don't know. */}
        <SegmentGroup.Root data-testid="segments" size={size as 'xs'} value="a">
          <SegmentGroup.Indicator />
          {['a', 'b'].map((value) => (
            <SegmentGroup.Item key={value} value={value}>
              <SegmentGroup.ItemHiddenInput />
              <SegmentGroup.ItemText>{value}</SegmentGroup.ItemText>
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
      </ChakraProvider>
    );
  });

  return {
    button: host.querySelector<HTMLElement>('[data-testid="button"]')!,
    segmentRoot: host.querySelector<HTMLElement>('[data-testid="segments"]')!,
  };
};

describe('segmentGroup slot recipe sizing', () => {
  // Segment groups sit beside buttons in toolbars; a size name must mean the
  // same outer height on both (Chakra's defaults run one size-name small).
  (['2xs', 'xs', 'sm', 'md'] as const).forEach((size) => {
    it(`matches the ${size} button height`, async () => {
      const { button, segmentRoot } = await renderPair(size);

      expect(segmentRoot.getBoundingClientRect().height).toBeCloseTo(button.getBoundingClientRect().height, 1);
    });
  });
});
