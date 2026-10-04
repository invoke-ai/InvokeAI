import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { isHotkeyModalLayerActive } from '@workbench/hotkeys/modalLayer';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { AlphaNoticeDialog } from './AlphaNoticeDialog';

const settings = vi.hoisted(() => ({
  snapshot: { preferences: { alphaNoticeAcknowledged: false }, status: 'idle' as 'idle' | 'ready' },
  listeners: new Set<() => void>(),
  patchWorkbenchPreferences: vi.fn(),
}));

vi.mock('@workbench/settings/store', async () => {
  const { useSyncExternalStore } = await import('react');

  return {
    patchWorkbenchPreferences: settings.patchWorkbenchPreferences,
    useWorkbenchSettingsSelector: (selector: (snapshot: typeof settings.snapshot) => unknown) =>
      useSyncExternalStore(
        (listener) => {
          settings.listeners.add(listener);
          return () => settings.listeners.delete(listener);
        },
        () => selector(settings.snapshot)
      ),
  };
});

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
  interpolation: { escapeValue: false },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const setSnapshot = (next: typeof settings.snapshot) =>
  act(() => {
    settings.snapshot = next;
    settings.listeners.forEach((listener) => listener());
  });

describe('AlphaNoticeDialog', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    settings.patchWorkbenchPreferences.mockReset();
    settings.snapshot = { preferences: { alphaNoticeAcknowledged: false }, status: 'idle' };
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <AlphaNoticeDialog />
          </I18nextProvider>
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('waits for preferences, then shows once with focus on the dismiss action, and records the acknowledgement', async () => {
    // Not before the account's preferences are known: an acknowledged account must not see it flash.
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();

    await setSnapshot({ preferences: { alphaNoticeAcknowledged: false }, status: 'ready' });
    await expect.element(page.getByRole('alertdialog', { name: i18n.t('alphaNotice.title') })).toBeVisible();
    await expect.element(page.getByRole('button', { name: i18n.t('alphaNotice.dismiss') })).toHaveFocus();
    expect(isHotkeyModalLayerActive()).toBe(true);
    expect(page.getByRole('link', { name: i18n.t('alphaNotice.reportLink') }).element()).toHaveAttribute(
      'rel',
      'noreferrer'
    );

    await page.getByRole('button', { name: i18n.t('alphaNotice.dismiss') }).click();
    expect(settings.patchWorkbenchPreferences).toHaveBeenCalledExactlyOnceWith({ alphaNoticeAcknowledged: true });

    await setSnapshot({ preferences: { alphaNoticeAcknowledged: true }, status: 'ready' });
    await expect.element(page.getByRole('alertdialog')).not.toBeInTheDocument();
    expect(isHotkeyModalLayerActive()).toBe(false);
  });

  it('stays closed for an account that already acknowledged it', async () => {
    await setSnapshot({ preferences: { alphaNoticeAcknowledged: true }, status: 'ready' });

    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(isHotkeyModalLayerActive()).toBe(false);
  });
});
