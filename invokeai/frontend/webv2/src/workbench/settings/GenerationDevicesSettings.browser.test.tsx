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
  updateGenerationDevices: vi.fn((_setting: unknown) => Promise.resolve()),
}));

vi.mock('@features/identity', () => ({
  useCapabilities: () => ({ canManageAppConfig: mocks.canManageAppConfig }),
}));

vi.mock('@features/queue/devices', async () => {
  const { getDeviceNameLabels } = await import('@features/queue/core/deviceLabels');

  return {
    getDeviceNameLabels,
    refreshGenerationDevices: () => mocks.refreshGenerationDevices(),
    updateGenerationDevices: (setting: unknown) => mocks.updateGenerationDevices(setting),
    useGenerationDevices: () => mocks.snapshot,
  };
});

import { GenerationDevicesSettings } from './GenerationDevicesSettings';

/**
 * Chakra's `Switch.HiddenInput` sits outside the viewport, so a real click cannot reach it — the visible control is the
 * click target.
 */
const switchControls = (): HTMLElement[] =>
  Array.from(host!.querySelectorAll<HTMLElement>('[data-scope="switch"][data-part="control"]'));

const TWO_GPUS = [
  { device: 'cuda:0', name: 'RTX 5090' },
  { device: 'cuda:1', name: 'RTX 4090' },
];

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  lng: 'en',
  interpolation: { escapeValue: false },
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (snapshot: Partial<GenerationDevicesSnapshot>): Promise<void> => {
  mocks.snapshot = {
    error: null,
    loadState: 'loaded',
    maxQueueHistory: undefined,
    options: [],
    setting: 'auto',
    ...snapshot,
  };

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerationDevicesSettings />
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

beforeEach(() => {
  mocks.canManageAppConfig = true;
  mocks.refreshGenerationDevices.mockClear();
  mocks.updateGenerationDevices.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerationDevicesSettings', () => {
  it('offers the auto switch with two GPUs and hides per-device switches while auto is on', async () => {
    await render({ options: TWO_GPUS, setting: 'auto' });

    expect(host?.textContent).toContain('Use every available accelerator');
    // Auto only shows its own switch; the per-device list stays collapsed.
    expect(host?.querySelectorAll('input[type="checkbox"]')).toHaveLength(1);
  });

  it('saves the devices in effect when auto is turned off, changing nothing else', async () => {
    await render({ options: TWO_GPUS, setting: 'auto' });

    await act(() => userEvent.click(switchControls()[0]!));

    // Leaving auto must not silently narrow the device set.
    expect(mocks.updateGenerationDevices).toHaveBeenCalledWith(['cuda:0', 'cuda:1']);
  });

  it('refuses to deselect the last GPU rather than letting the server 422', async () => {
    await render({ options: TWO_GPUS, setting: ['cuda:1'] });

    // Disabling the last selected device would persist an empty list rejected by the backend and startup.
    await act(() => userEvent.click(switchControls()[2]!));

    expect(mocks.updateGenerationDevices).not.toHaveBeenCalled();
    expect(host?.textContent).toContain('Select at least one generation device.');
  });
});
