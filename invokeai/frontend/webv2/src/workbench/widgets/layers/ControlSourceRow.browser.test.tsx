import type * as gallery from '@features/gallery';
import type { CanvasControlLayerContract, CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { ControlLayerIntent } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createControlLayer, createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ControlLayerSettings } from './ControlLayerSettings';

const harness = vi.hoisted(() => ({
  canvasUploads: [] as { category: unknown; project: unknown }[],
  galleryUploads: [] as File[],
  /** Holds the gallery upload until a test releases it. */
  release: null as (() => void) | null,
}));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof gallery>();
  return {
    ...actual,
    galleryTransfers: {
      ...actual.galleryTransfers,
      upload: async (file: File) => {
        harness.galleryUploads.push(file);
        await new Promise<void>((resolve) => {
          harness.release = resolve;
        });
        return { height: 48, imageName: 'uploaded.png', width: 80 };
      },
    },
  };
});
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  invalidateGallery: () => Promise.resolve(),
}));
vi.mock('@workbench/canvas-operations/backend/canvasImages', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  uploadCanvasImage: (_blob: Blob, options: { imageCategory?: string; projectId?: string }) => {
    harness.canvasUploads.push({ category: options.imageCategory, project: options.projectId });
    return Promise.resolve({ height: 32, imageName: 'bbox.png', width: 32 });
  },
}));
vi.mock('./useSelectedMainModel', () => ({ useSelectedMainModel: () => null }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'test-project',
  useOptionalWorkbenchCommands: () => null,
}));

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const PHOTO = { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' } as const;
const BBOX = { height: 512, width: 512, x: 40, y: 24 };
const empty = createControlLayer({ id: 'control-1', name: 'Control Layer 1' });
const filled: CanvasControlLayerContract = { ...empty, source: PHOTO };

type Engine = NonNullable<Parameters<typeof ControlLayerSettings>[0]['engine']>;

const intents: ControlLayerIntent[] = [];
let composite: { status: 'ok'; blob: Blob; rect: typeof BBOX } | { status: 'empty' } = { status: 'empty' };
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let engine: Engine | null = null;
const setTool = vi.fn();

const engineWith = (layers: CanvasLayerContract[]): Engine => {
  const document = { ...createEmptyCanvasDocument(), bbox: BBOX, stacks: stacksFrom(layers) };
  const model = createDocumentModel(document, { editRevision: 0, projectId: 'test-project' });
  const statuses = createControlDetectionStatusStore();
  const created = {
    document: { getDocument: () => document, model: () => model },
    exports: { exportRasterComposite: () => composite, hasExportableLayerContent: () => false },
    interaction: { get: () => false, subscribe: () => () => undefined },
    layers: { beginStructuralPreview: () => null, endStructuralPreview: () => undefined },
    lifecycle: { flushPendingUploads: () => Promise.resolve() },
    projectId: 'test-project',
    tools: { setTool },
  } as unknown as Engine;
  attachCanvasOperations(created, {
    applyControlLayerIntent: (_layerId: string, intent: ControlLayerIntent) => {
      intents.push(intent);
      return 'updated';
    },
    getControlDetectionStatus: statuses.get,
    subscribeControlDetectionStatus: statuses.subscribe,
  } as never);
  return created;
};

const queryClient = new QueryClient();
const view = (layer: CanvasControlLayerContract) => (
  <I18nextProvider i18n={i18n}>
    <ChakraProvider value={system}>
      <QueryClientProvider client={queryClient}>
        <ControlLayerSettings engine={engine} layer={layer} />
      </QueryClientProvider>
    </ChakraProvider>
  </I18nextProvider>
);
const settle = async (run: () => unknown = () => undefined) => {
  await act(async () => {
    await run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};
const render = async (layer: CanvasControlLayerContract) => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '320px';
  document.body.append(host);
  root = createRoot(host);
  engine = engineWith([layer]);
  await settle(() => root?.render(view(layer)));
};
/** The document now holds `layer`, as after the intent landed. */
const rerender = (layer: CanvasControlLayerContract) => settle(() => root?.render(view(layer)));

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  intents.length = 0;
  harness.canvasUploads.length = 0;
  harness.galleryUploads.length = 0;
  harness.release = null;
  composite = { status: 'empty' };
  setTool.mockClear();
  accountLifecycle.invalidate();
});

const button = (name: string) => page.getByRole('button', { exact: true, name });
const imageRow = () => page.getByRole('group', { name: 'Image' });

describe('Control layer Image row', () => {
  it('uploads a file, shows the upload in progress, and fills the layer at the bbox origin', async () => {
    await render(empty);
    const input = host!.querySelector<HTMLInputElement>('input[type="file"]')!;
    await act(async () => {
      await userEvent.upload(input, new File(['png'], 'portrait.png', { type: 'image/png' }));
    });
    await settle();

    expect(harness.galleryUploads.map((file) => file.name)).toEqual(['portrait.png']);
    // The pressed action shows its progress; the others wait for it.
    await expect.element(button('Upload')).toHaveAttribute('data-loading');
    await expect.element(button('Frame')).toBeDisabled();
    await expect.element(button('Draw')).toBeDisabled();

    await settle(() => harness.release?.());
    expect(intents).toEqual([
      {
        image: { height: 48, imageName: 'uploaded.png', width: 80 },
        kind: 'replace-image',
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 40, y: 24 },
      },
    ]);

    await rerender({
      ...empty,
      source: { image: { height: 48, imageName: 'uploaded.png', width: 80 }, type: 'image' },
    });
    expect(imageRow().element().textContent).toContain('portrait.png');
  });

  it('fills the layer from what is visible inside the frame, uploaded as a control image', async () => {
    composite = { blob: new Blob(['png']), rect: BBOX, status: 'ok' };
    await render(empty);
    await act(() => button('Frame').click());
    await settle();

    expect(harness.canvasUploads).toEqual([{ category: 'control', project: 'test-project' }]);
    expect(intents).toEqual([
      {
        image: { height: 32, imageName: 'bbox.png', width: 32 },
        kind: 'replace-image',
        transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 40, y: 24 },
      },
    ]);
    await rerender({ ...empty, source: { image: { height: 32, imageName: 'bbox.png', width: 32 }, type: 'image' } });
    expect(imageRow().element().textContent).toContain('From frame');
  });

  it('says so when nothing is visible inside the frame', async () => {
    const created = vi.spyOn(toaster, 'create');
    try {
      await render(empty);
      await act(() => button('Frame').click());
      await settle();

      expect(intents).toEqual([]);
      expect(created).toHaveBeenCalledWith(expect.objectContaining({ title: 'Nothing is visible inside the frame' }));
    } finally {
      created.mockRestore();
    }
  });

  it('drops an upload that finishes after the account changed, without a notice', async () => {
    accountLifecycle.activate('control-upload-a');
    const created = vi.spyOn(toaster, 'create');
    try {
      await render(empty);
      const input = host!.querySelector<HTMLInputElement>('input[type="file"]')!;
      await act(async () => {
        await userEvent.upload(input, new File(['png'], 'portrait.png', { type: 'image/png' }));
      });
      await settle();
      accountLifecycle.activate('control-upload-b');
      await settle(() => harness.release?.());

      expect(intents).toEqual([]);
      expect(created).not.toHaveBeenCalled();
      // The row is ready for the next action.
      await expect.element(button('Upload')).not.toHaveAttribute('data-loading');
    } finally {
      created.mockRestore();
    }
  });

  it('switches the layer to drawing and picks the brush', async () => {
    await render(empty);
    await act(() => button('Draw').click());

    expect(intents).toEqual([{ kind: 'draw' }]);
    expect(setTool).toHaveBeenCalledExactlyOnceWith('brush');
  });

  it('replaces a filled image from the keyboard and hands focus to the row that replaces the menu', async () => {
    await render(filled);
    expect(imageRow().element().textContent).toContain('Image · 96×64');
    const replace = button('Replace');
    await act(() => (replace.element() as HTMLElement).focus());
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.element(page.getByRole('menuitem', { name: 'Draw' })).toBeVisible();
    await act(() => userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}'));
    await settle();

    expect(intents).toEqual([{ kind: 'draw' }]);
    // Drawing empties the image: the empty actions take the place of the Replace menu, and focus with them.
    await rerender(empty);
    expect(imageRow().element().contains(document.activeElement)).toBe(true);
    expect(document.activeElement?.textContent).toBe('Upload');
  });
});
