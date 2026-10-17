import type { GenerationDevicesSnapshot } from '@features/queue/devices';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const mocks = vi.hoisted(() => ({
  canManageAppConfig: true,
  refreshGenerationDevices: vi.fn(() => Promise.resolve()),
  snapshot: null as GenerationDevicesSnapshot | null,
  updateMaxQueueHistory: vi.fn((_value: number | null) => Promise.resolve()),
}));

vi.mock('@features/identity', () => ({
  useCapabilities: () => ({ canManageAppConfig: mocks.canManageAppConfig }),
}));

vi.mock('@features/queue/devices', () => ({
  refreshGenerationDevices: () => mocks.refreshGenerationDevices(),
  updateMaxQueueHistory: (value: number | null) => mocks.updateMaxQueueHistory(value),
  useGenerationDevices: () => mocks.snapshot,
}));

import { MaxQueueHistorySettings } from './MaxQueueHistorySettings';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (snapshot: Partial<GenerationDevicesSnapshot>): Promise<void> => {
  mocks.snapshot = {
    error: null,
    loadState: 'loaded',
    maxQueueHistory: null,
    options: [],
    setting: 'auto',
    ...snapshot,
  };

  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <MaxQueueHistorySettings />
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

const limitInput = () => host.querySelector<HTMLInputElement>('input[aria-label="Queue history limit"]');
const button = (label: string) =>
  [...host.querySelectorAll('button')].find((candidate) => candidate.textContent === label)!;

const save = async (text: string) => {
  await userEvent.clear(limitInput()!);
  if (text) {
    await userEvent.type(limitInput()!, text);
  }
  await act(() => button('Save').click());
};

beforeEach(() => {
  mocks.canManageAppConfig = true;
  mocks.refreshGenerationDevices.mockClear();
  mocks.updateMaxQueueHistory.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('MaxQueueHistorySettings', () => {
  it('saves a cleared limit as "keep all" and 0 as "keep none"', async () => {
    await render({ maxQueueHistory: 5 });

    await save('');
    expect(mocks.updateMaxQueueHistory).toHaveBeenLastCalledWith(null);

    await save('0');
    expect(mocks.updateMaxQueueHistory).toHaveBeenLastCalledWith(0);
  });

  it('offers a retry instead of an editable "keep all" field while the server setting is unreadable', async () => {
    await render({ error: 'Failed to load generation devices', loadState: 'error', maxQueueHistory: undefined });

    expect(limitInput()).toBeNull();
    expect(host.textContent).toContain("Could not read the server's queue history setting.");

    expect(host.querySelector('[role="alert"]')?.textContent).toContain("Could not read the server's queue history");

    mocks.refreshGenerationDevices.mockImplementationOnce(() => {
      mocks.snapshot = { ...mocks.snapshot!, error: null, loadState: 'loaded', maxQueueHistory: 25 };
      return Promise.resolve();
    });
    const retry = button('Retry');
    retry.focus();
    await act(() => retry.click());
    await render({ maxQueueHistory: mocks.snapshot!.maxQueueHistory });

    expect(limitInput()?.value).toBe('25');
    // The Retry button is gone; focus lands on the restored field instead of falling to the page.
    expect(document.activeElement).toBe(limitInput());
  });

  it('still asks for a restart when the server accepted a limit but its answer was unreadable', async () => {
    mocks.updateMaxQueueHistory.mockImplementationOnce(() => {
      mocks.snapshot = { ...mocks.snapshot!, maxQueueHistory: undefined };
      return Promise.resolve();
    });
    await render({ maxQueueHistory: 5 });

    await save('10');
    await render({ maxQueueHistory: undefined });

    expect(host.textContent).toContain("Could not read the server's queue history setting.");
    expect(host.textContent).toContain('Restart InvokeAI for changes to take effect.');
  });

  it('keeps the limit editable when only the device listing failed', async () => {
    await render({ error: 'Failed to load generation devices', loadState: 'error', maxQueueHistory: 25 });

    expect(limitInput()?.value).toBe('25');
  });
});
