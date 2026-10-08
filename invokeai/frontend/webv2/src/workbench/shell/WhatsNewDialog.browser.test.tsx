import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { isHotkeyModalLayerActive } from '@workbench/hotkeys/modalLayer';
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

vi.mock('@platform/transport/http', () => ({
  apiFetchJson: vi.fn((path: string) =>
    path === '/api/v1/app/version'
      ? Promise.resolve({ version: '7.0.0-rc1' })
      : Promise.reject(new Error(`unexpected request ${path}`))
  ),
}));

const catalog = (await fetch('/locales/en.json').then((response) => response.json())) as {
  whatsNew: { highlights: Record<string, { description: string; title: string }>; items: string[] };
};

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

describe('WhatsNewDialog', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    settings.patchWorkbenchPreferences.mockReset();
    settings.snapshot = { preferences: { alphaNoticeAcknowledged: true, whatsNewSeenVersion: null }, status: 'idle' };
    // An account switch drops the previous test's request and dismissal; the version is not account data.
    accountLifecycle.invalidate();
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

  it('shows once for an unseen version after preferences load, then records the version on dismissal', async () => {
    await act(() => store.loadAppVersion());
    // Not before the account's preferences are known: an account that saw these notes must not see them flash.
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: '6.14.2' }));
    await expect.element(page.getByRole('dialog', { name: title() })).toBeVisible();
    await expect.element(page.getByText('v7.0.0-rc1')).toBeVisible();
    // An automatic open has no click behind it, so focus on a link would draw its keyboard ring; start on the panel.
    await expect.element(page.getByRole('dialog', { name: title() })).toHaveFocus();
    expect(isHotkeyModalLayerActive()).toBe(true);

    // Every catalogued highlight renders with its title, and the smaller notes list beneath them.
    for (const { description, title } of Object.values(catalog.whatsNew.highlights)) {
      await expect.element(page.getByText(title, { exact: true })).toBeVisible();
      await expect.element(page.getByText(description, { exact: true })).toBeVisible();
    }
    expect(
      page
        .getByRole('listitem')
        .elements()
        .map((item) => item.textContent)
    ).toEqual(catalog.whatsNew.items);

    expect(page.getByRole('link', { name: i18n.t('whatsNew.readReleaseNotes') }).element()).toHaveAttribute(
      'href',
      'https://github.com/invoke-ai/InvokeAI/releases/tag/v7.0.0-rc1'
    );
    expect(page.getByRole('link', { name: i18n.t('whatsNew.readTheDocs') }).element()).toHaveAttribute(
      'href',
      'https://v7.invoke.ai/'
    );

    await page.getByRole('button', { name: /close/i }).click();
    expect(settings.patchWorkbenchPreferences).toHaveBeenCalledExactlyOnceWith({ whatsNewSeenVersion: '7.0.0-rc1' });

    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: '7.0.0-rc1' }));
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    expect(isHotkeyModalLayerActive()).toBe(false);
  });

  it('waits for the alpha notice to be dismissed first', async () => {
    await act(() => store.loadAppVersion());
    await setSnapshot(ready({ alphaNoticeAcknowledged: false, whatsNewSeenVersion: null }));
    await expect.element(page.getByText('v7.0.0-rc1')).not.toBeInTheDocument();
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: null }));
    await expect.element(page.getByRole('dialog', { name: title() })).toBeVisible();
  });

  it('stays closed for a seen version until the app menu opens it, and reopening writes nothing', async () => {
    await act(() => store.loadAppVersion());
    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: '7.0.0-rc1' }));
    expect(document.querySelector('[role="dialog"]')).toBeNull();

    await act(() => store.openWhatsNew());
    await expect.element(page.getByRole('dialog', { name: title() })).toBeVisible();

    await page.getByRole('button', { name: /close/i }).click();
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    expect(settings.patchWorkbenchPreferences).not.toHaveBeenCalled();
  });

  it('shows an unseen version again for the next account after one dismissed it', async () => {
    await act(() => store.loadAppVersion());
    await setSnapshot(ready({ alphaNoticeAcknowledged: true, whatsNewSeenVersion: null }));
    await page.getByRole('button', { name: /close/i }).click();
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();

    await act(() => {
      accountLifecycle.invalidate();
    });
    await expect.element(page.getByRole('dialog', { name: title() })).toBeVisible();
  });
});
