import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ControlLayerSettings } from './ControlLayerSettings';

// Stable identities, as the real stores hand out, so only the table's arrival can re-read the policy.
const catalog = vi.hoisted(() => ({
  mainModel: { base: 'sd-1', key: 'sd1-main', name: 'SD 1.5', type: 'main' },
  models: [{ base: 'sd-1', key: 'sd1-controlnet', name: 'SD 1.5 ControlNet', type: 'controlnet' }],
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useModelsSelector: (selector: (snapshot: { models: typeof catalog.models }) => unknown) =>
    selector({ models: catalog.models }),
}));
vi.mock('./useSelectedMainModel', () => ({ useSelectedMainModel: () => catalog.mainModel }));

const getArchitectureCapabilities = vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>();
vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const layer = {
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: 'sd1-controlnet',
    weight: 1,
  },
  filter: null,
  id: 'control-1',
  isEnabled: true,
  isLocked: false,
  name: 'Control 1',
  opacity: 1,
  type: 'control',
  withTransparencyEffect: false,
} as unknown as CanvasControlLayerContract;

const ignoreOperationStarted = (): void => undefined;

const settle = async (run: () => void = () => undefined) => {
  await act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

const render = async () => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ControlLayerSettings engine={null} layer={layer} onOperationStarted={ignoreOperationStarted} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

/** The kind Select renders a hidden native select, so its offered kinds are readable without opening it. */
const offersKind = (kind: string) => host!.querySelector(`option[value="${kind}"]`) !== null;
const textOf = (role: 'alert' | 'status') =>
  [...host!.querySelectorAll(`[role="${role}"]`)].map((element) => element.textContent).join(' ');
const retryButton = () => [...host!.querySelectorAll('button')].find((button) => button.textContent === 'common.retry');

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  getArchitectureCapabilities.mockReset();
  // Returns the capability store and the core registry to their unloaded state between tests.
  accountLifecycle.invalidate();
});

describe('ControlLayerSettings and the capability table', () => {
  it('says the adapter is waiting for model capabilities rather than that it is unsupported', async () => {
    await render();

    expect(textOf('status')).toContain('widgets.layers.control.capabilitiesLoading');
  });

  it('keeps the retry and its focus through a retry that fails again, then offers the adapter kinds', async () => {
    // Capability arrival must update kinds even when the model base is unchanged.
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render();

    expect(textOf('alert')).toContain('widgets.layers.control.capabilitiesLoadFailed');
    expect(offersKind('controlnet')).toBe(false);

    // A retry that fails again: the button never unmounts, so a keyboard user's focus stays on it.
    let failLoad: (error: Error) => void = () => undefined;
    getArchitectureCapabilities.mockReturnValueOnce(
      new Promise((_, reject) => {
        failLoad = reject;
      })
    );
    await settle(() => {
      retryButton()?.focus();
      retryButton()?.click();
    });
    expect(document.activeElement).toBe(retryButton());
    expect(retryButton()?.getAttribute('aria-busy')).toBe('true');

    await settle(() => failLoad(new Error('Still out.')));
    expect(document.activeElement).toBe(retryButton());

    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(() => retryButton()?.click());

    expect(offersKind('controlnet')).toBe(true);
    expect(retryButton()).toBeUndefined();
    // The button that held focus is gone; focus moved into the panel instead of falling to <body>.
    expect(document.activeElement).not.toBe(document.body);
    expect(host!.contains(document.activeElement)).toBe(true);
  });

  it('leaves focus where the user moved it while the retry was in flight', async () => {
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render();

    let finishLoad: (rows: ArchitectureCapabilitiesRow[]) => void = () => undefined;
    getArchitectureCapabilities.mockReturnValueOnce(
      new Promise((resolve) => {
        finishLoad = resolve;
      })
    );
    await settle(() => {
      retryButton()?.focus();
      retryButton()?.click();
    });

    // Someone starts typing elsewhere; the load landing must not pull them into this panel.
    const elsewhere = document.createElement('input');
    document.body.append(elsewhere);
    elsewhere.focus();
    try {
      await settle(() => finishLoad(architectureCapabilitiesFixture));

      expect(retryButton()).toBeUndefined();
      expect(document.activeElement).toBe(elsewhere);
    } finally {
      elsewhere.remove();
    }
  });
});
