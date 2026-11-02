import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { RouterErrorScreen } from './RouterErrorScreen';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const originalClipboard = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
let root: Root | undefined;
let host: HTMLDivElement | undefined;

const stubClipboard = (writeText: (text: string) => Promise<void>) => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
};

const createError = () => {
  const error = new TypeError('Cannot read properties of undefined');
  error.stack = 'TypeError: Cannot read properties of undefined\n    at LayerRow (LayerRow.tsx:12:3)';
  return error;
};

const render = async (onGoHome = vi.fn()) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <RouterErrorScreen error={createError()} onGoHome={onGoHome} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return onGoHome;
};

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;

  if (originalClipboard) {
    Object.defineProperty(navigator, 'clipboard', originalClipboard);
  } else {
    Reflect.deleteProperty(navigator, 'clipboard');
  }
});

describe('RouterErrorScreen', () => {
  it('offers ways back and shows the error details on request', async () => {
    const onGoHome = await render();

    await expect.element(page.getByRole('heading', { name: 'Something went wrong' })).toBeVisible();
    expect(document.body.textContent).not.toContain('LayerRow.tsx');

    await userEvent.click(page.getByRole('button', { name: 'Show details' }));
    await expect.element(page.getByText(/at LayerRow \(LayerRow\.tsx:12:3\)/)).toBeVisible();
    await expect.element(page.getByRole('button', { name: 'Hide details' })).toHaveAttribute('aria-expanded', 'true');

    await userEvent.click(page.getByRole('button', { name: 'Go to Home' }));
    expect(onGoHome).toHaveBeenCalledOnce();
    await expect.element(page.getByRole('button', { name: 'Reload' })).toBeVisible();
  });

  it('copies the error and where it happened', async () => {
    const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue(undefined);
    stubClipboard(writeText);
    await render();

    await userEvent.click(page.getByRole('button', { name: 'Copy details' }));

    await expect.element(page.getByText('Details copied.')).toBeVisible();
    const copied = writeText.mock.calls[0]?.[0] ?? '';
    expect(copied).toContain('TypeError: Cannot read properties of undefined');
    expect(copied).toContain('at LayerRow (LayerRow.tsx:12:3)');
    expect(copied).toContain(`Location: ${window.location.href}`);
  });

  it('shows the details to copy by hand when the clipboard refuses', async () => {
    stubClipboard(vi.fn().mockRejectedValue(new Error('denied')));
    await render();

    await userEvent.click(page.getByRole('button', { name: 'Copy details' }));

    await expect.element(page.getByText(/Could not copy/)).toBeVisible();
    await expect.element(page.getByText(/at LayerRow \(LayerRow\.tsx:12:3\)/)).toBeVisible();
  });
});
