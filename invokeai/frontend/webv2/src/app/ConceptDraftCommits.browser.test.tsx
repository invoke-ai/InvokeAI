import type { GenerateLora } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { useUpscaleUi, type UpscaleUiAdapter } from '@features/upscale/ui/UpscaleUiContext';
import { UpscaleWidgetView } from '@features/upscale/ui/UpscaleWidgetView';
import { createDefaultVideoWidgetValues } from '@features/video';
import { useVideoUi, type VideoUiAdapter } from '@features/video/ui/VideoUiContext';
import { VideoWidgetView } from '@features/video/ui/VideoWidgetView';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { VideoUiAdapterProvider } from './VideoUiAdapter';

const model = (key: string, base: string, type: string, name = key): ModelConfig => ({
  base,
  file_size: 1,
  format: 'diffusers',
  hash: key,
  key,
  name,
  path: key,
  source: key,
  source_type: 'path',
  type,
});
const models = [
  { ...model('video-main', 'minimax-h3', 'main'), variant: 'fl2va' },
  model('video-first', 'minimax-h3', 'lora', 'MiniMax H3 Turbo LoRA'),
  model('video-second', 'minimax-h3', 'lora', 'Video Ink Wash'),
  model('upscale-main', 'sdxl', 'main'),
  model('upscale-first', 'sdxl', 'lora', 'Upscale Ink Wash'),
  model('upscale-second', 'sdxl', 'lora', 'Upscale Detail'),
];
const modelSnapshot = { models, status: 'loaded' };
const { notify } = vi.hoisted(() => ({ notify: vi.fn() }));
const noop = () => undefined;
const generationAdapter = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: noop },
} as unknown as GenerationUiAdapter;

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;
const captureAdapters = vi.fn<(adapters: { upscale: UpscaleUiAdapter; video: VideoUiAdapter }) => void>();

const CaptureAdapters = () => {
  captureAdapters({ upscale: useUpscaleUi(), video: useVideoUi() });
  return null;
};

vi.mock('react-i18next', () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
  getModelImageUrl: () => '',
  useModelsSelector: (selector: (snapshot: typeof modelSnapshot) => unknown) => selector(modelSnapshot),
  useOpenModelInManager: () => undefined,
}));
// Picker transport and prompt suggestions are unrelated to concept commits; retain the real list, rows, and controls.
vi.mock('@features/models/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ModelSelect: () => null,
}));
vi.mock('@features/gallery/mediaSlot', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryMediaSlot: () => null,
}));
vi.mock('@features/generation/components', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  NegativePromptField: () => null,
  PositivePromptField: () => null,
}));
vi.mock('@platform/ui/toaster', () => ({ toaster: { create: notify } }));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkbenchPreferenceSelector: () => false,
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => noop }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => noop }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Widget = 'upscale' | 'video';

const currentLoras = (widget: Widget): GenerateLora[] =>
  getProjectWidgetValues(store.getSnapshot().activeProject, widget).loras as GenerateLora[];

const render = async (widget: Widget) => {
  const loras = models
    .filter((candidate) => candidate.type === 'lora' && candidate.key.startsWith(widget))
    .map((candidate, index) => ({ isEnabled: true, model: candidate, weight: index === 0 ? 0.75 : 0.5 }));
  const defaults =
    widget === 'video' ? createDefaultVideoWidgetValues(models) : createDefaultUpscaleWidgetValues(models);
  store.commands.widgets.patchValues(widget, { ...defaults, loras, steps: 17 });
  const queryClient = new QueryClient();
  await act(() =>
    root.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={generationAdapter}>
            <VideoUiAdapterProvider>
              <UpscaleUiAdapterProvider>
                <CaptureAdapters />
                {widget === 'video' ? <VideoWidgetView /> : <UpscaleWidgetView />}
              </UpscaleUiAdapterProvider>
            </VideoUiAdapterProvider>
          </GenerationUiProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
  if (widget === 'upscale') {
    await act(() => {
      [...host.querySelectorAll<HTMLButtonElement>('button')]
        .find((button) => button.textContent === 'widgets.upscale.generation')
        ?.click();
    });
  }

  const rows = [...host.querySelectorAll<HTMLElement>('[role="listitem"]')];
  expect(rows).toHaveLength(2);
  return rows;
};

const step = (row: HTMLElement) =>
  row
    .querySelector('[role="slider"]')
    ?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));

beforeEach(() => {
  store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  notify.mockClear();
  captureAdapters.mockClear();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('concept drafts through widget adapters', () => {
  it.each(['upscale', 'video'] as const)(
    'flushes sibling %s drafts without losing same-turn settings changes',
    async (widget) => {
      const rows = await render(widget);
      await act(() => rows.forEach(step));
      expect(currentLoras(widget).map((lora) => lora.weight)).toEqual([0.75, 0.5]);

      await act(() => {
        store.commands.widgets.patchValues(widget, { steps: 23 });
        flushWorkbenchDrafts();
      });

      expect(currentLoras(widget).map((lora) => lora.weight)).toEqual([0.8, 0.55]);
      expect(getProjectWidgetValues(store.getSnapshot().activeProject, widget).steps).toBe(23);
    }
  );

  it.each(['upscale', 'video'] as const)(
    'keeps stale %s callbacks fenced to their originating project',
    async (widget) => {
      await render(widget);
      const patch = captureAdapters.mock.calls.at(-1)![0][widget].patchValues;
      const projectId = store.getSnapshot().activeProject.id;
      await act(() => store.commands.projects.create());
      const nextProjectValues = getProjectWidgetValues(store.getSnapshot().activeProject, widget);

      await act(() => patch((current: { steps: number }) => ({ steps: current.steps + 1 })));

      const originalProject = store.getSnapshot().projects.find((project) => project.id === projectId)!;
      expect(getProjectWidgetValues(originalProject, widget).steps).toBe(18);
      expect(getProjectWidgetValues(store.getSnapshot().activeProject, widget)).toBe(nextProjectValues);
    }
  );

  it.each(['upscale', 'video'] as const)(
    'does not restore a removed %s concept when another draft flushes',
    async (widget) => {
      const rows = await render(widget);
      await act(() => step(rows[1]!));
      await act(() => {
        rows[0]!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.removeConceptNamed"]')?.click();
        flushWorkbenchDrafts();
      });

      expect(currentLoras(widget).map((lora) => ({ key: lora.model.key, weight: lora.weight }))).toEqual([
        { key: `${widget}-second`, weight: 0.55 },
      ]);
      if (widget === 'video') {
        expect(getProjectWidgetValues(store.getSnapshot().activeProject, widget)).toMatchObject({
          acceleratorEnabled: false,
          acceleratorLoraKeys: [],
          steps: 50,
        });
        expect(notify).toHaveBeenCalledExactlyOnceWith({
          description: 'widgets.video.acceleratorBrokenDescription',
          title: 'widgets.video.acceleratorBroken',
          type: 'info',
        });
      }
    }
  );

  it.each(['upscale', 'video'] as const)(
    'writes only the removal when a %s row is removed with its own pending weight',
    async (widget) => {
      const rows = await render(widget);
      await act(() => step(rows[0]!));
      const projects = new Set<Project>();
      const unsubscribe = store.subscribe(() => projects.add(store.getSnapshot().activeProject));

      await act(() => {
        rows[0]!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.removeConceptNamed"]')?.click();
      });
      await act(() => flushWorkbenchDrafts());
      unsubscribe();

      expect(projects.size).toBe(1);
      expect(currentLoras(widget).map((lora) => lora.model.key)).toEqual([`${widget}-second`]);
    }
  );
});
