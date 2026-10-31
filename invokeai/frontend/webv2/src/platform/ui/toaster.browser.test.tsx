import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, createActionToast, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

await createInstance()
  .use(initReactI18next)
  .init({
    initAsync: false,
    lng: 'en',
    resources: { en: { translation: { common: { dismissNotification: 'Close this notice' } } } },
  });

const renderToaster = async () => {
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
};

const toastNamed = (title: string): HTMLElement | null =>
  [...document.querySelectorAll<HTMLElement>('[data-part="root"][data-scope="toast"]')].find(
    (toast) => toast.querySelector('[data-part="title"]')?.textContent === title
  ) ?? null;

const isGone = (title: string) => toastNamed(title)?.getAttribute('data-state') !== 'open';

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
    let id: string | null = null;
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

  it('keeps a toast without a timeout until its close button, while a timed one closes', async () => {
    await renderToaster();
    await act(() => {
      toaster.create({ duration: 200, title: 'Saved', type: 'success' });
      createActionToast({
        actions: [{ label: 'Retry', onClick: vi.fn() }],
        duration: Number.POSITIVE_INFINITY,
        title: 'Invocation failed',
        type: 'error',
      });
    });
    await expect.poll(() => toastNamed('Invocation failed')).not.toBeNull();

    await expect.poll(() => isGone('Saved'), { timeout: 3000 }).toBe(true);
    expect(toastNamed('Invocation failed')?.getAttribute('data-state')).toBe('open');

    const close = toastNamed('Invocation failed')!.querySelector<HTMLButtonElement>('[data-part="close-trigger"]')!;
    expect(close.getAttribute('aria-label')).toBe('Close this notice');
    await act(() => userEvent.click(close));
    await expect.poll(() => isGone('Invocation failed')).toBe(true);
  });

  it('dismisses an action toast when the scope its buttons act for ends', async () => {
    await renderToaster();
    const scope = new AbortController();
    await act(() => {
      createActionToast({
        actions: [{ label: 'Show in Gallery', onClick: vi.fn() }],
        duration: Number.POSITIVE_INFINITY,
        signal: scope.signal,
        title: 'Saved to Gallery',
        type: 'success',
      });
    });
    await expect.poll(() => toastNamed('Saved to Gallery')?.getAttribute('data-state')).toBe('open');

    await act(() => scope.abort());

    await expect.poll(() => isGone('Saved to Gallery')).toBe(true);

    // A scope that has already ended gets no toast at all.
    expect(
      createActionToast({ actions: [], signal: scope.signal, title: 'Saved too late', type: 'success' })
    ).toBeNull();
    expect(toaster.getVisibleToasts().some((toast) => toast.title === 'Saved too late')).toBe(false);
  });

  it('lets the keyboard reach and press a toast action from the toaster hotkey', async () => {
    await renderToaster();
    const onUndo = vi.fn();
    const editor = document.createElement('input');
    document.body.append(editor);
    editor.focus();
    await act(() => {
      createActionToast({
        actions: [{ label: 'Undo', onClick: onUndo }],
        duration: Number.POSITIVE_INFINITY,
        title: 'Moved 3 images',
        type: 'success',
      });
    });
    await expect.poll(() => toastNamed('Moved 3 images')?.getAttribute('data-state')).toBe('open');

    await userEvent.keyboard('{Alt>}t{/Alt}');
    await expect.poll(() => document.activeElement?.closest('[data-scope="toast"]')).not.toBeNull();
    // The hotkey focuses the region; Tab reaches the toast itself, then its first button.
    await userEvent.tab();
    await userEvent.tab();
    await expect.poll(() => document.activeElement?.textContent).toBe('Undo');
    await userEvent.keyboard('{Enter}');

    expect(onUndo).toHaveBeenCalledOnce();
    editor.remove();
  });
});
