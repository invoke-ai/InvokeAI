import type { CanvasInpaintMaskLayerContract } from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';

import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/models/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/models/core/architectureCapabilities.testing';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { commitPreparedEdit } from '@workbench/widgets/canvas/useStructuralCommit';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));
vi.mock('./useSelectedModelBase', () => ({
  useSelectedCfgScale: () => 7,
  useSelectedModelBase: () => 'sd-1',
}));

import { MaskGuidanceSettings } from './RegionalGuidanceSettings';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const GUIDANCE = { autoNegative: true, isEnabled: false, negativePrompt: 'blurry', positivePrompt: null };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let stub: StructuralEngineStub;

const liveLayer = () => getDocumentLayer(stub.document(), 'mask-1') as CanvasInpaintMaskLayerContract;

const Harness = () => {
  const layer = useSyncExternalStore(stub.subscribe, liveLayer);
  return layer.guidance ? (
    <MaskGuidanceSettings engine={stub.engine as never} guidance={layer.guidance} layerId={layer.id} />
  ) : null;
};

const render = async () => {
  setArchitectureCapabilities(architectureCapabilitiesFixture);
  stub = createStructuralEngineStub({
    layers: [layerContract('mask-1', 'inpaint_mask', { guidance: GUIDANCE } as never)],
  });
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
  vi.clearAllMocks();
});

const promptInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.positivePrompt' });

describe('MaskGuidanceSettings', () => {
  it('commits a typed prompt on blur as one whole-value guidance patch', async () => {
    await render();

    await act(() => promptInput().fill('a red door'));
    expect(stub.commits).toHaveLength(0);
    await act(() => userEvent.tab());

    expect(stub.commits).toHaveLength(1);
    expect(stub.commits[0]!.forward).toMatchObject({
      config: { guidance: { ...GUIDANCE, positivePrompt: 'a red door' }, layerType: 'inpaint_mask' },
    });
    await act(() => stub.engine.history.undo());
    expect(liveLayer().guidance).toEqual(GUIDANCE);
  });

  it('commits nothing when the field blurs after the guidance was removed', async () => {
    await render();
    await act(() => promptInput().fill('a red door'));
    const field = promptInput().element() as HTMLTextAreaElement;

    // The removal and the blur land before the editor re-renders without its guidance.
    await act(() => {
      commitPreparedEdit(stub.engine as never, 'Remove', (model) =>
        model.prepare({ config: { guidance: null, layerType: 'inpaint_mask' }, id: 'mask-1', type: 'patch-config' })
      );
      field.blur();
    });

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().guidance).toBeUndefined();
    expect(promptInput().query()).toBeNull();
  });
});
