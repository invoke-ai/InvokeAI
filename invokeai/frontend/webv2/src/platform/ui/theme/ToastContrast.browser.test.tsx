import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { getContrastRatio } from './contrastRatio.testing';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('toast contrast', () => {
  // Success and warning fills (green/orange 600) stay below AA even for full-strength white text.
  it.each(['error', 'info'] as const)('renders a %s description at AA contrast', async (type) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <AppToaster />
        </ChakraProvider>
      );
    });
    await act(() => {
      toaster.create({ description: 'Layer thumbnail rasterization failed.', title: 'Error', type });
    });
    await expect.poll(() => document.querySelector('[data-part="description"]')).not.toBeNull();
    const description = document.querySelector<HTMLElement>('[data-part="description"]')!;
    const toastRoot = description.closest<HTMLElement>('[data-part="root"]')!;
    const style = getComputedStyle(description);

    const ratio = getContrastRatio(style.color, getComputedStyle(toastRoot).backgroundColor, Number(style.opacity));

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});
