import type { GalleryImage } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { ModelConfig } from '@features/models';
import type { ControlLayerRejection } from '@workbench/widgets/canvas/invoke/prepareCanvasInvocation';

import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient } from '@tanstack/react-query';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';

import { useRegisterFirstPartyCommands } from './firstPartyCommands';

const handlers = vi.hoisted(() => new Map<string, () => unknown>());
const submitActiveInvocation = vi.hoisted(() =>
  vi.fn<(args: { formatControlLayerError: (rejection: ControlLayerRejection) => string }) => Promise<void>>(() =>
    Promise.resolve()
  )
);

// Generate values live in one mutable project so recall commits and typed edits land where the next read sees them.
const workbench = vi.hoisted(() => ({
  commands: { layout: {}, notifications: {}, queue: {}, widgets: {} } as Record<string, unknown>,
  generateValues: {} as Record<string, unknown>,
  queryClient: null as unknown,
}));
const recallFixtures = vi.hoisted(() => ({
  image: null as unknown,
  metadata: vi.fn(),
  models: [] as unknown[],
}));

vi.mock('@workbench/activeInvocationSubmission', () => ({ submitActiveInvocation }));
vi.mock('@workbench/image-actions/selectedImage', () => ({
  getSelectedGalleryImage: () => recallFixtures.image,
}));
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryImageMetadata: recallFixtures.metadata,
}));
vi.mock('@workbench/WorkbenchContext', () => {
  const getProject = () => ({
    id: 'project-1',
    invocation: { sourceId: 'generate' },
    widgetInstances: {
      generate: { state: { id: 'generate', label: 'generate', values: workbench.generateValues, version: 1 } },
    },
  });

  return {
    useWorkbenchCommands: () => workbench.commands,
    useWorkbenchExtensions: () => ({
      commands: {
        register: ({ handler, id }: { handler: () => unknown; id: string }) => {
          handlers.set(id, handler);
          return () => handlers.delete(id);
        },
      },
    }),
    useWorkbenchQueries: () => ({
      getProject: (projectId: string) => (projectId === 'project-1' ? getProject() : undefined),
      getSnapshot: () => ({ activeProject: getProject() }),
      isActiveProject: (projectId: string) => projectId === 'project-1',
    }),
  };
});
vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useQueryClient: () => workbench.queryClient,
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
  getModelsSnapshot: () => ({ models: recallFixtures.models, status: 'loaded' }),
}));

const english = createInstance();
const focus = createTestFocusController();
await english.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
const Commands = () => {
  useRegisterFirstPartyCommands();
  return null;
};

const renderCommands = async () => {
  const host = document.createElement('div');
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={english}>
        <FocusRegionProvider controller={focus}>
          <Commands />
        </FocusRegionProvider>
      </I18nextProvider>
    )
  );
};

afterEach(async () => {
  await act(() => root?.unmount());
  root = null;
  handlers.clear();
  submitActiveInvocation.mockClear();
  recallFixtures.metadata.mockReset();
  workbench.commands = { layout: {}, notifications: {}, queue: {}, widgets: {} };
  workbench.generateValues = {};
});

it('recalls from hotkeys on top of recalls and edits committed while the image record loaded', async () => {
  const model = {
    base: 'sdxl',
    file_size: 1,
    format: 'checkpoint',
    hash: 'hash',
    key: 'sdxl-model',
    name: 'SDXL',
    path: '/models/sdxl.safetensors',
    source: 'local',
    source_type: 'path',
    type: 'main',
  } as ModelConfig;
  const image: GalleryImage = {
    boardId: 'none',
    height: 768,
    imageCategory: 'general',
    imageName: 'selected.png',
    imageUrl: '/selected.png',
    queuedAt: '2026-06-19T00:00:00.000Z',
    sourceQueueItemId: 'backend-gallery',
    starred: false,
    thumbnailUrl: '/selected-thumb.png',
    width: 512,
  };
  let finishRead: ((metadata: unknown) => void) | undefined;

  accountLifecycle.activate('hotkey-recall-account');
  recallFixtures.image = image;
  recallFixtures.models = [model];
  recallFixtures.metadata.mockImplementation(
    () =>
      new Promise((resolve) => {
        finishRead = resolve;
      })
  );
  workbench.queryClient = new QueryClient();
  workbench.generateValues = { modelKey: model.key, positivePrompt: 'current prompt', steps: 20 };
  workbench.commands = {
    generation: {
      setSettings: (values: Record<string, unknown>) => {
        workbench.generateValues = values;
      },
    },
    layout: {},
    notifications: { add: vi.fn() },
    queue: {},
    widgets: { open: vi.fn() },
  };
  await renderCommands();

  let recalls: Promise<unknown>[] = [];
  await act(async () => {
    recalls = [handlers.get('viewer.recallPrompts')?.(), handlers.get('viewer.recallSeed')?.()] as Promise<unknown>[];
    await vi.waitFor(() => expect(finishRead).toBeDefined());
    // The second recall joins the shared read on a later turn.
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 0);
    });
  });
  expect(recallFixtures.metadata).toHaveBeenCalledOnce();
  // Typed while both recalls wait on the shared image record.
  workbench.generateValues = { ...workbench.generateValues, steps: 42 };
  await act(async () => {
    finishRead?.({ positive_prompt: 'recalled prompt', seed: 7 });
    await Promise.all(recalls);
  });

  expect(workbench.generateValues).toMatchObject({ positivePrompt: 'recalled prompt', seed: 7, steps: 42 });
});

it('words a blocked control layer from the locale when Invoke runs from its hotkey', async () => {
  await renderCommands();

  await act(async () => {
    await handlers.get('app.invoke')?.();
  });

  const [args] = submitActiveInvocation.mock.calls[0] ?? [];
  expect(args?.formatControlLayerError({ code: 'switch_adapter_kind', layerName: 'Sketch' })).toBe(
    'Control layer "Sketch": Choose a control type so a model for the selected main model can be picked.'
  );
});
