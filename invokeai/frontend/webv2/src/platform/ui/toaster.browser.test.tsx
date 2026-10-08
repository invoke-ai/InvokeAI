import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

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

describe('AppToaster', () => {
  it('wraps unbroken model sources inside the toast', async () => {
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
      toaster.create({
        description:
          'stabilityai/stable-diffusion-xl-base-1.0::sd_xl_base_1.0_0.9vae_with_a_very_long_filename.safetensors',
        title: 'Model installed',
        type: 'success',
      });
    });
    await expect.poll(() => document.querySelector('[data-part="description"]')).not.toBeNull();
    const description = document.querySelector<HTMLElement>('[data-part="description"]')!;
    const toastRoot = description.closest<HTMLElement>('[data-part="root"]')!;

    expect(description.getBoundingClientRect().right).toBeLessThanOrEqual(toastRoot.getBoundingClientRect().right);
    expect(description.scrollWidth).toBeLessThanOrEqual(description.clientWidth);
  });
});
