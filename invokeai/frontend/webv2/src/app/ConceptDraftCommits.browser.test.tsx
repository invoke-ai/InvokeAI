import type { GenerateLora } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
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
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useState, useSyncExternalStore, type ReactNode } from 'react';
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
  { ...model('unknown-h3', 'minimax-h3', 'main', 'Unknown H3'), format: 'checkpoint', variant: 'novel_task' },
  model('video-first', 'minimax-h3', 'lora', 'MiniMax H3 Turbo LoRA'),
  model('video-second', 'minimax-h3', 'lora', 'Video Ink Wash'),
  model('upscale-main', 'sdxl', 'main'),
  model('upscale-first', 'sdxl', 'lora', 'Upscale Ink Wash'),
  model('upscale-second', 'sdxl', 'lora', 'Upscale Detail'),
  // Not listed by `render`: only offered through the concept picker.
  model('picked-video', 'minimax-h3', 'lora'),
  model('picked-upscale', 'sdxl', 'lora'),
];
const modelSnapshot = { models, status: 'loaded' };
const { notify, picker } = vi.hoisted(() => ({ notify: vi.fn(), picker: { pick: null as unknown } }));
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
// The concept picker hands over `picker.pick` unfiltered, as a stale picker would.
vi.mock('@features/models/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ModelSelect: ({
    modelTypes,
    onChange,
    pickerHeader,
    onPickerClose,
  }: {
    modelTypes?: readonly string[];
    onChange: (model: unknown) => void;
    pickerHeader?: ReactNode;
    onPickerClose?: () => void;
  }) => {
    const [open, setOpen] = useState(false);
    if (modelTypes?.includes('lora')) {
      return (
        <button
          aria-label="Add concept"
          data-testid="concept-picker"
          type="button"
          onClick={() => onChange(picker.pick)}
        />
      );
    }
    if (!modelTypes?.includes('main')) {
      return null;
    }
    return (
      <>
        <button type="button" data-testid="open-model" onClick={() => setOpen(true)}>
          Open
        </button>
        {open && (
          <div>
            {pickerHeader}
            <button
              type="button"
              data-testid="dismiss-model"
              onClick={() => {
                setOpen(false);
                onPickerClose?.();
              }}
            >
              Cancel
            </button>
            <button
              type="button"
              data-testid="pick-unknown"
              onClick={() => {
                onChange(models.find((item) => item.key === 'unknown-h3'));
                setOpen(false);
                onPickerClose?.();
              }}
            >
              Unknown H3
            </button>
          </div>
        )}
      </>
    );
  },
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
vi.mock('@platform/ui/toaster', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  toaster: { create: notify },
}));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkbenchPreferenceSelector: () => false,
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => noop }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => noop }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) => {
    const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot);
    return selector(getResolvedProjectWidgetValues(snapshot.activeProject, widgetId, snapshot.scopedDefaults));
  },
  useOptionalWorkbenchCommands: () => store.commands,
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
    .filter((candidate) => candidate.type === 'lora' && candidate.key.startsWith(`${widget}-`))
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
            <DndContext>
              <VideoUiAdapterProvider>
                <UpscaleUiAdapterProvider>
                  <CaptureAdapters />
                  {widget === 'video' ? <VideoWidgetView /> : <UpscaleWidgetView />}
                </UpscaleUiAdapterProvider>
              </VideoUiAdapterProvider>
            </DndContext>
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

  it('applies the accelerator rule when a video concept is removed beside a pending draft', async () => {
    const rows = await render('video');
    await act(() => step(rows[1]!));
    await act(() => {
      rows[0]!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.removeConceptNamed"]')?.click();
      flushWorkbenchDrafts();
    });

    expect(currentLoras('video').map((lora) => ({ key: lora.model.key, weight: lora.weight }))).toEqual([
      { key: 'video-second', weight: 0.55 },
    ]);
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video')).toMatchObject({
      acceleratorEnabled: false,
      acceleratorLoraKeys: [],
      steps: 50,
    });
    expect(notify).toHaveBeenCalledExactlyOnceWith({
      description: 'widgets.video.acceleratorBrokenDescription',
      title: 'widgets.video.acceleratorBroken',
      type: 'info',
    });
  });

  it.each(['upscale', 'video'] as const)(
    'writes only the removal when a %s row is double-clicked away with its own pending weight',
    async (widget) => {
      const rows = await render(widget);
      await act(() => step(rows[0]!));
      const projects = new Set<Project>();
      const unsubscribe = store.subscribe(() => projects.add(store.getSnapshot().activeProject));

      await act(() => {
        const remove = rows[0]!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.removeConceptNamed"]');
        remove?.click();
        remove?.click();
      });
      await act(() => flushWorkbenchDrafts());
      unsubscribe();

      expect(projects.size).toBe(1);
      expect(currentLoras(widget).map((lora) => lora.model.key)).toEqual([`${widget}-second`]);
    }
  );

  it.each([
    ['upscale', 'picked-upscale', 'picked-video'],
    ['video', 'picked-video', 'picked-upscale'],
  ] as const)(
    'adds a picked %s concept once, and only when its model can load it',
    async (widget, compatible, incompatible) => {
      await render(widget);
      const pick = (key: string) =>
        act(() => {
          picker.pick = models.find((candidate) => candidate.key === key);
          host.querySelector<HTMLButtonElement>('[data-testid="concept-picker"]')?.click();
        });

      await pick(compatible);
      expect(currentLoras(widget).map((lora) => [lora.model.key, lora.weight])).toEqual([
        [`${widget}-first`, 0.75],
        [`${widget}-second`, 0.5],
        [compatible, 0.75],
      ]);

      const project = store.getSnapshot().activeProject;
      await pick(compatible);
      await pick(`${widget}-first`);
      await pick(incompatible);
      expect(store.getSnapshot().activeProject).toBe(project);
    }
  );
  it('resets the picker choice on dismiss and selects unknown detected H3 as FL2VA', async () => {
    await render('video');
    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="open-model"]')!.click());
    const ref = [...host.querySelectorAll('button')].find((button) => button.textContent === 'Ref2VA')!;
    await act(() => ref.click());
    expect(ref.getAttribute('aria-pressed')).toBe('true');
    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="dismiss-model"]')!.click());
    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="open-model"]')!.click());
    expect(
      [...host.querySelectorAll('button')]
        .find((button) => button.textContent === 'widgets.video.h3Detected')
        ?.getAttribute('aria-pressed')
    ).toBe('true');
    await act(() => host.querySelector<HTMLButtonElement>('[data-testid="pick-unknown"]')!.click());
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video')).toMatchObject({
      h3SelectedTask: 'fl2va',
      h3ModeOverride: false,
      model: { variant: 'fl2va' },
    });
  });

  it('flushes pending concept edits and preserves a same-turn edit when switching H3 task', async () => {
    await render('video');
    const checkpoint = models.find((item) => item.key === 'unknown-h3')!;
    await act(() =>
      store.commands.widgets.patchValues('video', {
        model: checkpoint,
        modelKey: checkpoint.key,
        h3SelectedTask: 'fl2va',
        componentSourceModel: models[0],
      })
    );
    const currentRows = [...host.querySelectorAll<HTMLElement>('[role="listitem"]')];
    expect(currentRows.length).toBe(2);
    await act(() => step(currentRows[1]!));
    await act(() => {
      store.commands.widgets.patchValues('video', { positivePrompt: 'newest prompt' });
      const label = [...host.querySelectorAll('label')].find((item) =>
        item.textContent?.includes('widgets.video.h3SwitchMode')
      );
      expect(label).toBeTruthy();
      label!.querySelector<HTMLInputElement>('input')!.click();
    });
    const current = getProjectWidgetValues(store.getSnapshot().activeProject, 'video');
    expect(current).toMatchObject({
      positivePrompt: 'newest prompt',
      h3SelectedTask: 'ref2va',
      h3ModeOverride: true,
      acceleratorEnabled: false,
    });
    expect(currentLoras('video').find((lora) => lora.model.key === 'video-second')?.weight).toBe(0.55);
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ description: 'widgets.video.h3TurboRemoved' }));
  });
  it('blocks a stale switch callback when a reference was just added', async () => {
    await render('video');
    const checkpoint = models.find((item) => item.key === 'unknown-h3')!;
    await act(() =>
      store.commands.widgets.patchValues('video', {
        model: checkpoint,
        modelKey: checkpoint.key,
        h3SelectedTask: 'fl2va',
        componentSourceModel: models[0],
      })
    );
    const reference = { kind: 'image', detail: 'match', image: { image_name: 'new.png', width: 512, height: 512 } };
    await act(() => {
      store.commands.widgets.patchValues('video', { references: [reference] });
      const label = [...host.querySelectorAll('label')].find((item) =>
        item.textContent?.includes('widgets.video.h3SwitchMode')
      )!;
      label.querySelector<HTMLInputElement>('input')!.click();
    });
    expect(getProjectWidgetValues(store.getSnapshot().activeProject, 'video')).toMatchObject({
      h3SelectedTask: 'fl2va',
      references: [reference],
    });
    expect(notify).toHaveBeenCalledWith(expect.objectContaining({ title: 'widgets.video.h3ClearModeInputs' }));
  });
});
