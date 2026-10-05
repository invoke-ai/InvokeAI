import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, createActionToast, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

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

  it('offers its actions as matching buttons, each dismissing the toast before it acts', async () => {
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
    const calls: string[] = [];
    const dismiss = vi.spyOn(toaster, 'dismiss').mockImplementation((toastId) => {
      calls.push(`dismiss:${String(toastId)}`);
    });
    let id = '';
    await act(() => {
      id = createActionToast({
        actions: [
          { label: 'Retry', onClick: vi.fn() },
          { label: 'Show in Gallery', onClick: () => calls.push('show') },
        ],
        title: 'Saved to Gallery, but not to its board',
        type: 'warning',
      });
    });
    await expect.poll(() => document.querySelectorAll('[data-part="root"][data-scope="toast"] button').length).toBe(3);
    const buttons = [...document.querySelectorAll<HTMLButtonElement>('[data-part="root"][data-scope="toast"] button')];
    const retry = buttons.find((button) => button.textContent === 'Retry')!;
    const show = buttons.find((button) => button.textContent === 'Show in Gallery')!;

    expect(getComputedStyle(show).fontSize).toBe(getComputedStyle(retry).fontSize);
    expect(show.getBoundingClientRect().height).toBe(retry.getBoundingClientRect().height);

    await act(() => show.click());

    expect(calls).toEqual([`dismiss:${id}`, 'show']);
    dismiss.mockRestore();
  });
});
