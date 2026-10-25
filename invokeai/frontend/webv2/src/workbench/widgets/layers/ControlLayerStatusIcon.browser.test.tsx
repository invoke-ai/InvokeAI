import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type { CanvasControlLayerContract } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ControlLayerStatusIcon } from './ControlLayerStatusIcon';

// Stable identities, as the real stores hand out: a fresh array or model per render would change an
// input React Compiler memoises on, and re-read the table by accident rather than by subscription.
const catalog = vi.hoisted(() => ({
  mainModel: { base: 'sd-1', key: 'sd1-main', name: 'SD 1.5', type: 'main' },
  models: [
    { base: 'sd-1', key: 'sd1-controlnet', name: 'SD 1.5 ControlNet', type: 'controlnet' },
    { base: 'sd-1', key: 'sd1-depth', name: 'SD 1.5 Depth', type: 'controlnet' },
  ],
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
  id: 'control-1',
  isEnabled: true,
  isLocked: false,
  name: 'Control 1',
  opacity: 1,
  source: { image: { height: 64, imageName: 'control.png', width: 64 }, type: 'image' },
  type: 'control',
} as unknown as CanvasControlLayerContract;

/** Typed Depth with a detector whose map has not been produced yet: the map is stale. */
const pendingDepthLayer = {
  ...layer,
  adapter: { ...layer.adapter, model: 'sd1-depth' },
  controlType: 'depth',
  detection: { settings: {}, type: 'depth_anything_depth_estimation' },
} as CanvasControlLayerContract;

const statuses = createControlDetectionStatusStore();
const engine = {};
attachCanvasOperations(engine, {
  getControlDetectionStatus: statuses.get,
  subscribeControlDetectionStatus: statuses.subscribe,
} as never);

const settle = async (run: () => void = () => undefined) => {
  await act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

const render = async (shown: CanvasControlLayerContract = layer) => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ControlLayerStatusIcon engine={engine} layer={shown} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const warning = () => host!.querySelector('[aria-label^="widgets.layers.control.needsAttention"]');

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  getArchitectureCapabilities.mockReset();
  statuses.clear();
  i18n.removeResourceBundle('en', 'translation');
  // Returns the capability store and the core registry to their unloaded state between tests.
  accountLifecycle.invalidate();
});

describe('ControlLayerStatusIcon and the capability table', () => {
  it('flags nothing about a valid adapter while the table is still loading', async () => {
    // Missing capabilities must not label a valid adapter unsupported.
    await render();

    expect(warning()).toBeNull();
  });

  it('points at the failed capability load, and clears once a retry succeeds, without remounting', async () => {
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render();

    const label = warning()?.getAttribute('aria-label') ?? '';
    expect(label).toContain('widgets.layers.control.capabilitiesLoadFailed');
    expect(label).not.toContain('unsupported_adapter');

    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(ensureArchitectureCapabilitiesLoaded);

    expect(warning()).toBeNull();
  });
});

describe('ControlLayerStatusIcon detection and new reasons', () => {
  it('shows a labelled spinner while the layer detects, then the warning detection left behind', async () => {
    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render(pendingDepthLayer);

    // A stale map blocks Invoke, so the row says so until detection catches up.
    expect(warning()?.getAttribute('aria-label')).toContain('widgets.layers.control.validation.detection_pending');

    await settle(() =>
      statuses.set('control-1', { detector: 'depth_anything_depth_estimation', status: 'processing' })
    );

    expect(warning()).toBeNull();
    expect(host!.querySelector('.chakra-spinner')).not.toBeNull();
    expect(host!.textContent).toBe('widgets.layers.control.detecting');

    await settle(() => statuses.set('control-1', { status: 'idle' }));

    expect(host!.querySelector('.chakra-spinner')).toBeNull();
    expect(warning()).not.toBeNull();
  });

  it('shows the spinner, not the pending warning, while detection waits to run', async () => {
    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(ensureArchitectureCapabilitiesLoaded);
    i18n.addResource('en', 'translation', 'widgets.layers.control.detectionQueued', 'Waiting to run {{detector}}…');
    i18n.addResource(
      'en',
      'translation',
      'widgets.layers.control.filters.depth_anything_depth_estimation',
      'Depth Anything'
    );
    await render(pendingDepthLayer);

    await settle(() => statuses.set('control-1', { status: 'queued' }));

    expect(warning()).toBeNull();
    expect(host!.querySelector('.chakra-spinner')).not.toBeNull();
    // The queued status names no detector; the label uses the one the layer asks for, and does not claim it runs.
    expect(host!.textContent).toBe('Waiting to run Depth Anything…');
  });
});
