import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { WhatsNewDialog } from './WhatsNewDialog';
import * as store from './whatsNewStore';

type Snapshot = {
  preferences: { alphaNoticeAcknowledged: boolean; whatsNewSeenVersion: string | null };
  status: 'idle' | 'ready';
};

const settings = vi.hoisted(() => ({
  snapshot: {
    preferences: { alphaNoticeAcknowledged: true, whatsNewSeenVersion: null },
    status: 'idle',
  } as Snapshot,
  listeners: new Set<() => void>(),
  patchWorkbenchPreferences: vi.fn(),
}));

vi.mock('@workbench/settings/store', async () => {
  const { useSyncExternalStore } = await import('react');

  return {
    patchWorkbenchPreferences: settings.patchWorkbenchPreferences,
    useWorkbenchSettingsSelector: (selector: (snapshot: Snapshot) => unknown) =>
      useSyncExternalStore(
        (listener) => {
          settings.listeners.add(listener);
          return () => settings.listeners.delete(listener);
        },
        () => selector(settings.snapshot)
      ),
  };
});

const http = vi.hoisted(() => ({ resolveVersion: (_value: { version: string }) => {} }));

// The version read stays pending until the test resolves it.
vi.mock('@platform/transport/http', () => ({
  apiFetchJson: vi.fn((path: string) =>
    path === '/api/v1/app/version'
      ? new Promise((resolve) => {
          http.resolveVersion = resolve;
        })
      : Promise.reject(new Error(`unexpected request ${path}`))
  ),
}));

const catalog = (await fetch('/locales/en.json').then((response) => response.json())) as Record<string, unknown>;

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: catalog } },
  interpolation: { escapeValue: false },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const setSnapshot = (next: Snapshot) =>
  act(() => {
    settings.snapshot = next;
    settings.listeners.forEach((listener) => listener());
  });

const ready = (preferences: Snapshot['preferences']): Snapshot => ({ preferences, status: 'ready' });

const title = () => i18n.t('whatsNew.whatsNewInInvoke');

/** Its own file: the store keeps the version for the page's lifetime, and this case needs it still unknown. */
describe('WhatsNewDialog with a slow version read', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            <WhatsNewDialog />
          </I18nextProvider>
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('does not reopen notes closed by hand when the version arrives late', async () => {
    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: '6.14.2' }));

    // The boot read is still pending when the menu opens the notes.
    void store.loadAppVersion();
    await act(() => store.openWhatsNew());
    await expect.element(page.getByRole('dialog', { name: title() })).toBeVisible();
    await page.getByRole('button', { name: /close/i }).click();
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();

    await act(async () => {
      http.resolveVersion({ version: '7.0.0-rc1' });
      await store.loadAppVersion();
    });
    expect(store.getWhatsNewSnapshot().version).toBe('7.0.0-rc1');
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    expect(settings.patchWorkbenchPreferences).not.toHaveBeenCalled();
  });
});
