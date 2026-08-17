/* oxlint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { StarterModel, StarterModelBundle } from '@features/models/core/types';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BundleChips } from './BundleChips';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const makeModel = (name: string): StarterModel => ({
  base: 'sdxl',
  description: '',
  dependencies: null,
  format: 'checkpoint',
  is_installed: false,
  name,
  source: `https://example.test/${name}`,
  type: 'main',
});

// A dozen long bundle names guarantee the row overflows a narrow host no
// matter what font metrics the browser resolves.
const bundles: StarterModelBundle[] = Array.from({ length: 12 }, (_, index) => ({
  models: [makeModel(`model-${index}`)],
  name: `Very Long Bundle Name ${index}`,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('BundleChips overflow', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    host.style.cssText = 'width:240px;';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('scrolls the chip row horizontally and keeps trailing content outside the scroll region', async () => {
    await act(async () => {
      root.render(
        <ChakraProvider value={system}>
          <BundleChips
            bundles={bundles}
            selectedName={null}
            starterCount={40}
            trailing={<Box data-testid="trailing-marker">trailing</Box>}
            onSelect={() => undefined}
          />
        </ChakraProvider>
      );
      await Promise.resolve();
    });

    const viewport = host.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]');
    const horizontalScrollbar = host.querySelector<HTMLElement>(
      '[data-scope="scroll-area"][data-part="scrollbar"][data-orientation="horizontal"]'
    );
    const trailing = host.querySelector<HTMLElement>('[data-testid="trailing-marker"]');

    expect(viewport).not.toBeNull();
    expect(horizontalScrollbar).not.toBeNull();
    expect(trailing).not.toBeNull();

    // (a) the chip row overflows the viewport horizontally.
    expect(viewport!.scrollWidth).toBeGreaterThan(viewport!.clientWidth);

    // (b) a horizontal scrollbar part is rendered (asserted above via the
    // non-null query; also confirm zag hasn't hidden it via data attrs).
    expect(horizontalScrollbar!.getAttribute('data-orientation')).toBe('horizontal');

    // (c) the trailing element is not part of the scrolling content — it
    // lives outside the viewport entirely, so it never scrolls with the chips.
    expect(viewport!.contains(trailing)).toBe(false);
  });
});
