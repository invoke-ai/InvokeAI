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
 * Chakra's `Switch.HiddenInput` sits outside the viewport, so a real click cannot
 * reach it — the visible control is the click target.
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
  resources: { en: { translation: { settingsDialog: { changedSetting: '{{setting}} differs from its default' } } } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (snapshot: Partial<GenerationDevicesSnapshot>): Promise<void> => {
  mocks.snapshot = { error: null, loadState: 'loaded', options: [], setting: 'auto', ...snapshot };

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
  it('explains that parallel generation needs more than one accelerator on a single-device box', async () => {
    await render({ options: [{ device: 'cuda:0', name: 'RTX 5090' }] });

    // One GPU in auto mode has no device choice to expose.
    expect(host?.textContent).toContain('RTX 5090');
    expect(host?.textContent).toContain('needs more than one accelerator');
    expect(host?.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);
  });

  it('says so when no accelerator is present', async () => {
    await render({ options: [] });

    expect(host?.textContent).toContain('No accelerators were detected');
  });

  it('offers the auto switch with two GPUs and hides per-device switches while auto is on', async () => {
    await render({ options: TWO_GPUS, setting: 'auto' });

    expect(host?.textContent).toContain('Use every available accelerator');
    // Auto only shows its own switch; the per-device list stays collapsed.
    expect(host?.querySelectorAll('input[type="checkbox"]')).toHaveLength(1);
  });

  it('lists a switch per device when a specific set is configured', async () => {
    await render({ options: TWO_GPUS, setting: ['cuda:0'] });

    expect(host?.textContent).toContain('RTX 5090');
    expect(host?.textContent).toContain('RTX 4090');
    // The auto switch plus one per device.
    expect(host?.querySelectorAll('input[type="checkbox"]')).toHaveLength(3);
  });

  it('marks explicit device lists as changed and removes the marker when auto is restored', async () => {
    await render({ options: TWO_GPUS, setting: ['cuda:0', 'cuda:1'] });
    expect(host?.querySelectorAll('[role="img"][aria-label$="differs from its default"]')).toHaveLength(1);
    await render({ options: TWO_GPUS, setting: 'auto' });
    expect(host?.querySelector('[role="img"][aria-label$="differs from its default"]')).toBeNull();
    await render({ loadState: 'loading', options: TWO_GPUS, setting: ['cuda:0'] });
    expect(host?.querySelector('[role="img"][aria-label$="differs from its default"]')).toBeNull();
  });

  it('saves the devices in effect when auto is turned off, changing nothing else', async () => {
    await render({ options: TWO_GPUS, setting: 'auto' });

    await act(() => userEvent.click(switchControls()[0]!));

    // Leaving auto must not silently narrow the device set.
    expect(mocks.updateGenerationDevices).toHaveBeenCalledWith(['cuda:0', 'cuda:1']);
  });

  it('shows the restart notice only after a change is saved', async () => {
    await render({ options: TWO_GPUS, setting: 'auto' });

    expect(host?.textContent).not.toContain('Restart InvokeAI');

    await act(() => userEvent.click(switchControls()[0]!));

    expect(host?.textContent).toContain('Restart InvokeAI for changes to take effect.');
  });

  it('refuses to deselect the last GPU rather than letting the server 422', async () => {
    await render({ options: TWO_GPUS, setting: ['cuda:1'] });

    // Disabling the last selected device would persist an empty list rejected by the backend and startup.
    await act(() => userEvent.click(switchControls()[2]!));

    expect(mocks.updateGenerationDevices).not.toHaveBeenCalled();
    expect(host?.textContent).toContain('Select at least one generation device.');
  });

  it('shows a read-only summary to a non-admin', async () => {
    mocks.canManageAppConfig = false;
    await render({ options: TWO_GPUS, setting: ['cuda:1'] });

    expect(host?.textContent).toContain('RTX 4090');
    expect(host?.textContent).toContain('Only an administrator can change which accelerators are used.');
    expect(host?.querySelectorAll('input[type="checkbox"]')).toHaveLength(0);
    expect(
      host?.querySelector('[role="img"][aria-label="Generation devices differs from its default"]')
    ).not.toBeNull();
  });

  it('surfaces a load failure instead of rendering empty controls', async () => {
    await render({ error: 'Failed to load generation devices', loadState: 'error' });

    expect(host?.textContent).toContain('Failed to load generation devices');
  });
});
