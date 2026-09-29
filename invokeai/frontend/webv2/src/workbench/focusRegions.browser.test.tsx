import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('./settings/store', () => ({
  useWorkbenchPreferenceSelector: (selector: (preferences: { showFocusRegionHighlight: boolean }) => unknown) =>
    selector({ showFocusRegionHighlight: true }),
}));

import { FocusRegionProvider, useFocusRegionProps } from './focusRegions';

const FocusableRegion = () => <Box data-testid="focus-region" h="20" {...useFocusRegionProps('center')} />;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('focus region highlight', () => {
  it('lays the highlight over the borders beside the region', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <FocusRegionProvider>
            <FocusableRegion />
          </FocusRegionProvider>
        </ChakraProvider>
      );
    });

    const region = host.querySelector<HTMLElement>('[data-testid="focus-region"]');
    expect(region).not.toBeNull();

    await act(() => region?.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true })));

    expect(region?.getAttribute('data-highlighted')).toBe('true');
    const highlight = getComputedStyle(region!, '::after');
    expect([highlight.top, highlight.right, highlight.bottom, highlight.left]).toEqual(['0px', '-1px', '0px', '-1px']);
  });
});
