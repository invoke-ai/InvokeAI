import type {
  CanvasInpaintMaskLayerContract,
  CanvasMaskFillContract,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
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

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));

import { InpaintMaskSettings } from './InpaintMaskSettings';

type Engine = ComponentProps<typeof InpaintMaskSettings>['engine'];

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** The structural engine over the real reducer and history; previews flush synchronously, as a settled frame would. */
let stub: StructuralEngineStub;
let sampleRequests = 0;

const documentLayer = (): CanvasInpaintMaskLayerContract =>
  getDocumentLayer(stub.document(), 'mask-1') as CanvasInpaintMaskLayerContract;
const commits = () => stub.commits;

/** Lands a fill edit from elsewhere (a script, the tint editor) as a recorded step. */
const commitFillFromElsewhere = (fill: Partial<CanvasMaskFillContract>): void => {
  const outcome = commitPreparedEdit(stub.engine, 'Fill', (model) => {
    const live = model.getLayer('mask-1') as CanvasInpaintMaskLayerContract;
    return model.prepare({
      config: { layerType: 'inpaint_mask', mask: { fill: { ...live.mask.fill, ...fill } } },
      id: 'mask-1',
      type: 'patch-config',
    });
  });
  expect(outcome).toEqual({ status: 'committed' });
};

/** The stub's structural half plus the sampler the picker's eyedropper calls; built once per render. */
let engine: Engine;
const engineWithSampler = (): Engine =>
  ({
    ...stub.engine,
    tools: {
      requestColorSample: () => {
        sampleRequests += 1;
        return Promise.resolve('#123456');
      },
    },
  }) as unknown as Engine;

const Harness = () => {
  const layer = useSyncExternalStore(stub.subscribe, documentLayer);
  return <InpaintMaskSettings engine={engine} layer={layer} />;
};

const settle = (action: () => void = () => undefined): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const render = async () => {
  stub = createStructuralEngineStub({
    layers: [
      layerContract('mask-1', 'inpaint_mask', {
        mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
        name: 'Inpaint Mask',
      }),
    ],
    schedulePreview: (flush) => {
      flush();
      return () => undefined;
    },
  });
  engine = engineWithSampler();
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);

  await settle(() => {
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
  sampleRequests = 0;
  await settle(() => root?.unmount());
  document.querySelectorAll('[data-scope="color-picker"][data-part="positioner"]').forEach((el) => el.remove());
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

const fillOf = (edit: PreparedDocumentEdit, side: 'forward' | 'inverse'): CanvasMaskFillContract =>
  (edit[side] as unknown as { config: { mask: { fill: CanvasMaskFillContract } } }).config.mask.fill;

const openPicker = async () => {
  const trigger = host!.querySelector<HTMLElement>('[aria-label="widgets.layers.maskFill.color"]')!;
  await settle(() => trigger.click());
};

/** Presses on the picker's color area, which previews the color under the pointer until the release commits it. */
const pressArea = async () => {
  const area = document.querySelector<HTMLElement>('[data-scope="color-picker"][data-part="area"]')!;
  const rect = area.getBoundingClientRect();
  const at = (fraction: number) => ({
    clientX: rect.left + rect.width * fraction,
    clientY: rect.top + rect.height * fraction,
  });
  const pointer = (target: EventTarget, type: string, fraction: number) =>
    settle(() =>
      target.dispatchEvent(
        new PointerEvent(type, { bubbles: true, button: 0, isPrimary: true, pointerId: 1, ...at(fraction) })
      )
    );
  await pointer(area, 'pointerdown', 0.2);
  return { release: () => pointer(document, 'pointerup', 0.2) };
};

describe('mask fill eyedropper', () => {
  it('samples the canvas through the engine and commits the picked fill color once', async () => {
    await render();
    await openPicker();

    // With an engine present the picker offers the canvas sampler, not the screen eyedropper.
    const sampleButton = document.querySelector<HTMLElement>('[aria-label="common.colorPicker.sampleFromCanvas"]');
    expect(sampleButton).not.toBeNull();
    await settle(() => sampleButton!.click());

    expect(sampleRequests).toBe(1);
    expect(commits()).toHaveLength(1);
    expect(fillOf(commits()[0]!, 'forward')).toEqual({ color: '#123456', style: 'solid' });
    expect(fillOf(commits()[0]!, 'inverse')).toEqual({ color: '#ff0000', style: 'solid' });
  });
});

describe('mask fill color gesture', () => {
  it('previews the color under the pointer and records the gesture once from the fill it started on', async () => {
    await render();
    await openPicker();
    const drag = await pressArea();

    const previewed = documentLayer().mask.fill.color;
    expect(previewed).not.toBe('#ff0000');
    expect(commits()).toHaveLength(0);

    await drag.release();

    expect(commits()).toHaveLength(1);
    expect(fillOf(commits()[0]!, 'forward').color).toBe(previewed);
    expect(fillOf(commits()[0]!, 'inverse')).toEqual({ color: '#ff0000', style: 'solid' });
    await stub.engine.history.undo();
    expect(documentLayer().mask.fill.color).toBe('#ff0000');
  });

  it('keeps an undo that lands mid-gesture: the picker follows the undone fill and the release records nothing over it', async () => {
    await render();
    commitFillFromElsewhere({ color: '#00ff00' });
    await settle();
    await openPicker();
    const drag = await pressArea();
    expect(documentLayer().mask.fill.color).not.toBe('#00ff00');

    await act(() => stub.engine.history.undo());

    expect(documentLayer().mask.fill.color).toBe('#ff0000');

    await drag.release();

    expect(documentLayer().mask.fill.color).toBe('#ff0000');
    expect(commits()).toHaveLength(1);
    expect(stub.history.canUndo()).toBe(false);
    expect(stub.history.canRedo()).toBe(true);
    expect(notify.error).not.toHaveBeenCalled();
  });
});
