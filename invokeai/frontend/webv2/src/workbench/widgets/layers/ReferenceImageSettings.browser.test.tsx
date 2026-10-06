import type {
  CanvasRegionalGuidanceLayerContract,
  PreparedDocumentEdit,
  RegionalGuidanceReferenceImage,
} from '@workbench/canvas-engine/api';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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
vi.mock('@workbench/WorkbenchContext', () => ({
  useOptionalWorkbenchCommands: () => null,
  useWorkbenchCommands: () => ({ notifications: { reportError: () => undefined } }),
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useModelsSelector: (selector: (snapshot: { models: [] }) => unknown) => selector({ models: [] }),
}));
vi.mock('@features/gallery/picker', () => ({
  GalleryPickerPopover: ({ children }: { children: ReactNode }) => children,
}));
vi.mock('./useSelectedModelBase', () => ({ useSelectedModelBase: () => null }));

import { ReferenceImageSettings } from './ReferenceImageSettings';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Engine = NonNullable<Parameters<typeof ReferenceImageSettings>[0]['engine']>;

const commits: PreparedDocumentEdit[] = [];

/** The layer as the engine holds it; the one stable engine reads it live, as the real engine's model does. */
let documentLayer: CanvasRegionalGuidanceLayerContract;
const layerListeners = new Set<() => void>();
const subscribeLayer = (listener: () => void) => {
  layerListeners.add(listener);
  return () => layerListeners.delete(listener);
};
const publishLayer = (next: CanvasRegionalGuidanceLayerContract): void => {
  documentLayer = next;
  layerListeners.forEach((listener) => listener());
};

const engine = {
  document: {
    model: () =>
      createDocumentModel(documentFrom([documentLayer], documentLayer.id), { editRevision: 0, projectId: 'p' }),
  },
  layers: {
    endStructuralPreview: () => undefined,
    commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
      commits.push(edit);
      const { config } = edit.forward as unknown as { config: { referenceImages: RegionalGuidanceReferenceImage[] } };
      publishLayer({ ...documentLayer, referenceImages: config.referenceImages });
      return { status: 'committed' as const };
    },
  },
} as unknown as Engine;

const ipAdapterRef = (id: string): RegionalGuidanceReferenceImage => ({
  config: {
    beginEndStepPct: [0, 1],
    clipVisionModel: 'ViT-H',
    image: null,
    method: 'full',
    model: null,
    type: 'ip_adapter',
    weight: 1,
  },
  id,
  isEnabled: true,
});

const Harness = () => {
  const layer = useSyncExternalStore(subscribeLayer, () => documentLayer);
  return <ReferenceImageSettings engine={engine} layer={layer} refId="ref-1" />;
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
  documentLayer = layerContract('rg-1', 'regional_guidance', {
    referenceImages: [ipAdapterRef('ref-1'), ipAdapterRef('ref-2')],
  } as never) as CanvasRegionalGuidanceLayerContract;
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root!.render(
      <QueryClientProvider client={new QueryClient()}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <DndContext>
              <Harness />
            </DndContext>
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    )
  );
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  commits.length = 0;
  vi.clearAllMocks();
});

const weightSlider = () => page.getByRole('slider', { name: 'widgets.layers.regionalGuidance.weight' });
const refsOf = (edit: PreparedDocumentEdit, side: 'forward' | 'inverse') =>
  (edit[side] as unknown as { config: { referenceImages: RegionalGuidanceReferenceImage[] } }).config.referenceImages;

/** Starts a drag on the weight scrubber; offsets are fractions of its -1..2 track from the press. */
const startDrag = async () => {
  const frame = weightSlider().element().closest<HTMLElement>('[data-scope="scrubber"]')!;
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

describe('ReferenceImageSettings weight', () => {
  it('shows a drag locally and records it once when it ends', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.05);
    await drag.moveTo(0.1);

    await expect.element(weightSlider()).toHaveAttribute('aria-valuetext', '1.30');
    expect(commits).toHaveLength(0);

    await drag.release();

    expect(commits).toHaveLength(1);
    expect(refsOf(commits[0]!, 'forward').map((ref) => ref.config)).toMatchObject([{ weight: 1.3 }, { weight: 1 }]);
    expect(refsOf(commits[0]!, 'inverse').map((ref) => ref.config)).toMatchObject([{ weight: 1 }, { weight: 1 }]);
  });

  it('records nothing when a drag returns to where it started', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.1);
    await drag.moveTo(0);
    await drag.release();

    expect(commits).toHaveLength(0);
  });

  it('keeps an image that lands mid-drag, in the commit and in its undo', async () => {
    await render();
    const drag = await startDrag();
    await drag.moveTo(0.1);
    // An upload started earlier finishes while the weight is still held.
    const image = { height: 64, imageName: 'upload.png', thumbnailUrl: 'about:blank', url: 'about:blank', width: 64 };
    await settle(() =>
      publishLayer({
        ...documentLayer,
        referenceImages: documentLayer.referenceImages.map((ref) =>
          ref.id === 'ref-1'
            ? ({ ...ref, config: { ...ref.config, image } } as unknown as RegionalGuidanceReferenceImage)
            : ref
        ),
      })
    );
    await drag.release();

    expect(commits).toHaveLength(1);
    expect(refsOf(commits[0]!, 'forward')[0]!.config).toMatchObject({ image, weight: 1.3 });
    expect(refsOf(commits[0]!, 'inverse')[0]!.config).toMatchObject({ image, weight: 1 });
  });
});
