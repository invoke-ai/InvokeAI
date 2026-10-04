/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider, Dialog } from '@chakra-ui/react';
import { system } from '@theme/system';
import { isHotkeyModalLayerActive } from '@workbench/hotkeys/modalLayer';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { SettingsDialogHost } from './SettingsDialogHost';
import { openWorkbenchSettings, settingsDialogStore } from './settingsDialogStore';

const initialLoad = vi.hoisted(() => {
  let reject!: (error: Error) => void;
  const promise = new Promise<never>((_resolve, rejectPromise) => {
    reject = rejectPromise;
  });
  let resolveRetry!: () => void;
  const retryPromise = new Promise<void>((resolve) => {
    resolveRetry = resolve;
  });
  return { promise, reject, retryPromise, resolveRetry };
});

vi.mock('./dialogResource', async () => {
  const { createDeferredResource } = await import('@workbench/deferredResource');
  let failedOnce = false;
  return {
    settingsDialogResource: createDeferredResource(() => {
      if (!failedOnce) {
        failedOnce = true;
        return initialLoad.promise;
      }
      return initialLoad.retryPromise.then(() => ({
        default: () => (
          <Dialog.Body>
            <Dialog.Title>Settings: Behavior</Dialog.Title>
            <input aria-label="Search settings" data-settings-search />
            <button type="button">Reset all</button>
          </Dialog.Body>
        ),
      }));
    }),
  };
});

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
  interpolation: { escapeValue: false },
});

let host: HTMLDivElement | undefined;
let root: Root | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  settingsDialogStore.patchSnapshot({ isOpen: false, returnFocus: null });
});

it('keeps a named modal through loading, failure, retry, and closing before restoring focus', async () => {
  await page.viewport(1280, 900);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host, {
    onCaughtError: (error) => {
      if (!(error instanceof Error) || error.message !== 'Settings chunk unavailable') {
        throw error;
      }
    },
  });
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <button type="button" onClick={() => openWorkbenchSettings('behavior')}>
            Open settings
          </button>
          <SettingsDialogHost />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  expect(isHotkeyModalLayerActive()).toBe(false);
  const trigger = page.getByRole('button', { name: 'Open settings', exact: true });
  await act(() => trigger.click());
  await expect.element(page.getByRole('dialog', { name: 'Settings', exact: true })).toBeVisible();
  await expect.element(page.getByRole('status')).toHaveTextContent(i18n.t('common.loading'));
  expect(isHotkeyModalLayerActive()).toBe(true);

  await act(() => initialLoad.reject(new Error('Settings chunk unavailable')));
  await expect.element(page.getByRole('alert')).toMatchTextContent(i18n.t('settingsDialog.loadFailed'));
  await expect.element(page.getByRole('dialog', { name: 'Settings', exact: true })).toBeVisible();
  const retry = page.getByRole('button', { name: i18n.t('common.retry'), exact: true });
  await expect.element(retry).toBeEnabled();
  expect(isHotkeyModalLayerActive()).toBe(true);
  await act(() => retry.click());
  await expect.element(page.getByRole('alert')).toHaveAttribute('aria-busy', 'true');
  expect(isHotkeyModalLayerActive()).toBe(true);
  await act(() => initialLoad.resolveRetry());

  await expect.element(page.getByRole('dialog', { name: 'Settings: Behavior', exact: true })).toBeVisible();
  await expect.element(page.getByRole('alert')).not.toBeInTheDocument();
  expect(isHotkeyModalLayerActive()).toBe(true);
  // `/` from a control inside the dialog jumps to search; inside the search it types.
  await page.getByRole('button', { name: 'Reset all', exact: true }).click();
  await userEvent.keyboard('/');
  await expect.element(page.getByRole('textbox', { name: 'Search settings', exact: true })).toHaveFocus();
  await userEvent.keyboard('/');
  await expect.element(page.getByRole('textbox', { name: 'Search settings', exact: true })).toHaveValue('/');
  await act(() =>
    document.querySelector<HTMLButtonElement>('[data-scope="dialog"][data-part="close-trigger"]')!.click()
  );
  expect(document.querySelector('[data-scope="dialog"][data-part="content"][data-state="closed"]')).not.toBeNull();
  expect(isHotkeyModalLayerActive()).toBe(true);
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
  await expect.element(trigger).toHaveFocus();
  expect(isHotkeyModalLayerActive()).toBe(false);
});
