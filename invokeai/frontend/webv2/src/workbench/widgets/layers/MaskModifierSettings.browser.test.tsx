import type { CanvasInpaintMaskLayerContract, PreparedDocumentEdit } from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';

import { ChakraProvider } from '@chakra-ui/react';
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
import { page } from 'vitest/browser';

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));

import { MaskModifierSettings } from './MaskModifierSettings';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Engine = NonNullable<Parameters<typeof MaskModifierSettings>[0]['engine']>;

/** The structural engine over the real reducer and history; previews flush synchronously, as a settled frame would. */
let stub: StructuralEngineStub;
const documentLayer = (): CanvasInpaintMaskLayerContract =>
  getDocumentLayer(stub.document(), 'mask-1') as CanvasInpaintMaskLayerContract;

type Noise = NonNullable<CanvasInpaintMaskLayerContract['noise']>;

/** Lands a noise edit from elsewhere (a tree toggle, a script) as a recorded step, prepared from the live layer. */
const commitNoiseFromElsewhere = (update: (noise: Noise) => Noise): void => {
  const outcome = commitPreparedEdit(stub.engine as never, 'Noise', (model) => {
    const live = model.getLayer('mask-1') as CanvasInpaintMaskLayerContract;
    return model.prepare({
      config: { layerType: 'inpaint_mask', noise: update(live.noise!) },
      id: 'mask-1',
      type: 'patch-config',
    });
  });
  expect(outcome).toEqual({ status: 'committed' });
};

const Harness = () => {
  const layer = useSyncExternalStore(stub.subscribe, documentLayer);
  return <MaskModifierSettings engine={stub.engine as unknown as Engine} kind="mask-noise" layer={layer} />;
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const settle = (run: () => void = () => undefined) =>
  act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });

const render = async () => {
  stub = createStructuralEngineStub({
    layers: [layerContract('mask-1', 'inpaint_mask', { noise: { isEnabled: true, level: 0.25 } } as never)],
    schedulePreview: (flush) => {
      flush();
      return () => undefined;
    },
  });
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

const noiseSlider = () => page.getByRole('slider', { name: 'widgets.layers.maskFill.noiseLevel' });
const noiseOf = (edit: PreparedDocumentEdit, side: 'forward' | 'inverse') =>
  (edit[side] as unknown as { config: { noise: { isEnabled: boolean; level: number } } }).config.noise;
const commits = () => stub.commits;

/** Starts a drag on the noise scrubber; `moveTo` takes track fractions from the press, `release` ends it there. */
const startDrag = async () => {
  const frame = noiseSlider().element().closest<HTMLElement>('[data-scope="scrubber"]')!;
  const rect = frame.getBoundingClientRect();
  const startX = rect.left + rect.width / 2;
  const xAt = (offset: number) => startX + offset * (rect.width - 20);
  const pointer = (target: EventTarget, type: string, x: number) =>
    settle(() => target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: x, pointerId: 1 })));
  await pointer(frame, 'pointerdown', startX);
  let last = 0;
  return {
    moveTo: async (offset: number) => {
      last = offset;
      await pointer(window, 'pointermove', xAt(offset));
    },
    release: () => pointer(window, 'pointerup', xAt(last)),
  };
};

describe('MaskModifierSettings', () => {
  it('scrubs in whole percent and records the drag as one step from where it started', async () => {
    await render();
    await expect.element(noiseSlider()).toHaveAttribute('aria-valuetext', '25%');

    const drag = await startDrag();
    await drag.moveTo(0.1);
    await drag.moveTo(0.2);

    expect(documentLayer().noise).toEqual({ isEnabled: true, level: 0.45 });
    expect(commits()).toHaveLength(0);

    await drag.release();

    expect(commits()).toHaveLength(1);
    expect(noiseOf(commits()[0]!, 'forward')).toEqual({ isEnabled: true, level: 0.45 });
    expect(noiseOf(commits()[0]!, 'inverse')).toEqual({ isEnabled: true, level: 0.25 });
    await stub.engine.history.undo();
    expect(documentLayer().noise).toEqual({ isEnabled: true, level: 0.25 });
  });

  it('keeps a toggle that lands mid-drag in the previews, the commit, and its undo', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.1);
    await settle(() => commitNoiseFromElsewhere((noise) => ({ ...noise, isEnabled: false })));

    // The toggle ended the previewed gesture, restoring its level before it landed.
    expect(documentLayer().noise).toEqual({ isEnabled: false, level: 0.25 });

    await drag.moveTo(0.3);

    expect(documentLayer().noise).toEqual({ isEnabled: false, level: 0.55 });

    await drag.release();

    expect(noiseOf(commits().at(-1)!, 'forward')).toEqual({ isEnabled: false, level: 0.55 });
    expect(noiseOf(commits().at(-1)!, 'inverse')).toEqual({ isEnabled: false, level: 0.25 });
    await stub.engine.history.undo();
    expect(documentLayer().noise).toEqual({ isEnabled: false, level: 0.25 });
  });

  it('keeps an undo that lands mid-drag and records the rest of the drag from the undone level', async () => {
    await render();
    commitNoiseFromElsewhere((noise) => ({ ...noise, level: 0.5 }));
    await settle();
    const drag = await startDrag();
    await drag.moveTo(0.1);
    expect(documentLayer().noise?.level).toBeCloseTo(0.6, 5);

    await act(() => stub.engine.history.undo());

    expect(documentLayer().noise).toEqual({ isEnabled: true, level: 0.25 });
    expect(stub.history.canRedo()).toBe(true);

    await drag.moveTo(0.2);
    await drag.release();

    expect(noiseOf(commits().at(-1)!, 'inverse')).toEqual({ isEnabled: true, level: 0.25 });
    expect(stub.history.canRedo()).toBe(false);
    await stub.engine.history.undo();
    expect(documentLayer().noise).toEqual({ isEnabled: true, level: 0.25 });
    expect(notify.error).not.toHaveBeenCalled();
  });

  it('records nothing and restores the level when a drag returns to where it started', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.2);
    await drag.moveTo(0);
    await drag.release();

    expect(commits()).toHaveLength(0);
    expect(documentLayer().noise).toEqual({ isEnabled: true, level: 0.25 });
    expect(stub.history.canUndo()).toBe(false);
  });
});
