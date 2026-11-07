import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import * as LocalRunListModule from './LocalRunList';
import { createTestLocalRun, createTestQueueUiAdapter } from './queueUiAdapter.testing';
import { QueueUiProvider } from './QueueUiContext';
import { createLocalRunListLoader, LocalRunsSection, type LocalRunListLoader } from './QueueWidgetView';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

const runs = [createTestLocalRun({ label: 'Waiting run' })];

const render = (rows: LocalRunListLoader) =>
  act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueueUiProvider adapter={createTestQueueUiAdapter()}>
            <button type="button">Elsewhere in the widget</button>
            <LocalRunsSection filter="all" rows={rows} runs={runs} />
          </QueueUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

it('keeps a failed chunk failed until Retry, which imports it once more and shows the rows', async () => {
  // The chunk fails as it does with the server down, then loads once the server is back.
  const importer = vi
    .fn<() => Promise<typeof LocalRunListModule>>()
    .mockRejectedValueOnce(new TypeError('Failed to fetch dynamically imported module'))
    .mockResolvedValue(LocalRunListModule);
  const rows = createLocalRunListLoader(importer);

  rows.preload();
  await vi.waitFor(() => expect(importer).toHaveBeenCalledOnce());
  // Neither another preload nor rendering refetches the failed chunk.
  rows.preload();
  await render(rows);

  const alert = await vi.waitUntil(() => host.querySelector<HTMLElement>('[role="alert"]'));
  expect(alert.textContent).toContain("Couldn't load the runs waiting to be submitted.");
  expect(host.textContent).toContain('Submitting');
  expect(host.querySelector('button')?.textContent).toBe('Elsewhere in the widget');
  expect(importer).toHaveBeenCalledOnce();

  await act(() => [...alert.querySelectorAll('button')].find((button) => button.textContent === 'Retry')!.click());

  await expect
    .poll(() => host.querySelector('[role="list"][aria-label="Submitting"]')?.textContent)
    .toContain('Waiting run');
  expect(importer).toHaveBeenCalledTimes(2);
  expect(host.querySelector('[role="alert"]')).toBeNull();
});
