import type {
  CanvasInpaintMaskLayerContract,
  CanvasLayerPreviewMutation,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import { documentFrom, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
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

const commits: PreparedDocumentEdit[] = [];
const previews: CanvasLayerPreviewMutation[] = [];

/** The layer as the engine holds it; the one stable engine reads it live, as the real engine's model does. */
let documentLayer: CanvasInpaintMaskLayerContract;
const layerListeners = new Set<() => void>();
const subscribeLayer = (listener: () => void) => {
  layerListeners.add(listener);
  return () => layerListeners.delete(listener);
};
const publishLayer = (next: CanvasInpaintMaskLayerContract): void => {
  documentLayer = next;
  layerListeners.forEach((listener) => listener());
};

/** Applies a config mutation the way the reducer does: the named fields replace the layer's. */
const applyConfig = (mutation: unknown): void => {
  const { config } = mutation as { config: Record<string, unknown> };
  const { layerType: _layerType, ...fields } = config;
  publishLayer({ ...documentLayer, ...fields } as CanvasInpaintMaskLayerContract);
};

const engine = {
  document: {
    model: () =>
      createDocumentModel(documentFrom([documentLayer], documentLayer.id), { editRevision: 0, projectId: 'p' }),
  },
  layers: {
    beginStructuralPreview: () => ({
      apply: (action: CanvasLayerPreviewMutation) => {
        previews.push(action);
        applyConfig(action);
        return true;
      },
      cancel: (restore?: CanvasLayerPreviewMutation) => {
        if (restore) {
          applyConfig(restore);
        }
      },
      commit: (_label: string, edit: PreparedDocumentEdit) => {
        commits.push(edit);
        applyConfig(edit.forward);
        return { status: 'committed' as const };
      },
    }),
    commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
      commits.push(edit);
      applyConfig(edit.forward);
      return { status: 'committed' as const };
    },
  },
} as unknown as Engine;

const Harness = () => {
  const layer = useSyncExternalStore(subscribeLayer, () => documentLayer);
  return <MaskModifierSettings engine={engine} kind="mask-noise" layer={layer} />;
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
  documentLayer = layerContract('mask-1', 'inpaint_mask', {
    noise: { isEnabled: true, level: 0.25 },
  } as never) as CanvasInpaintMaskLayerContract;
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
  commits.length = 0;
  previews.length = 0;
  vi.clearAllMocks();
});

const noiseSlider = () => page.getByRole('slider', { name: 'widgets.layers.maskFill.noiseLevel' });
const noiseOf = (edit: PreparedDocumentEdit, side: 'forward' | 'inverse') =>
  (edit[side] as unknown as { config: { noise: { isEnabled: boolean; level: number } } }).config.noise;

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

    expect(documentLayer.noise).toEqual({ isEnabled: true, level: 0.45 });
    expect(commits).toHaveLength(0);

    await drag.release();

    expect(commits).toHaveLength(1);
    expect(noiseOf(commits[0]!, 'forward')).toEqual({ isEnabled: true, level: 0.45 });
    expect(noiseOf(commits[0]!, 'inverse')).toEqual({ isEnabled: true, level: 0.25 });
  });

  it('keeps a toggle that lands mid-drag in the previews, the commit, and its undo', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.1);
    await settle(() => publishLayer({ ...documentLayer, noise: { ...documentLayer.noise!, isEnabled: false } }));
    await drag.moveTo(0.3);

    expect(previews.at(-1)).toMatchObject({ config: { noise: { isEnabled: false, level: 0.55 } } });

    await drag.release();

    expect(noiseOf(commits[0]!, 'forward')).toEqual({ isEnabled: false, level: 0.55 });
    expect(noiseOf(commits[0]!, 'inverse')).toEqual({ isEnabled: false, level: 0.25 });
  });

  it('records nothing and restores the level when a drag returns to where it started', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.2);
    await drag.moveTo(0);
    await drag.release();

    expect(commits).toHaveLength(0);
    expect(documentLayer.noise).toEqual({ isEnabled: true, level: 0.25 });
  });
});
