import type { PromptHistoryItem } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { UpscaleWidgetView } from '@features/upscale/ui/UpscaleWidgetView';
import { createDefaultVideoWidgetValues } from '@features/video';
import { VideoWidgetView } from '@features/video/ui/VideoWidgetView';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { runNotificationUndo } from '@workbench/shell/notifications/undoRegistry';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';
import { VideoUiAdapterProvider } from './VideoUiAdapter';

const model = (key: string, base: string, type: string, extra: Partial<ModelConfig> = {}): ModelConfig => ({
  base,
  file_size: 1,
  format: 'gguf_quantized',
  hash: key,
  key,
  name: key,
  path: key,
  source: key,
  source_type: 'path',
  type,
  ...extra,
});
// Wan reads a negative prompt; the recall Undo exists for exactly that field.
const models = [model('video-main', 'wan', 'main', { variant: 't2v_a14b' }), model('upscale-main', 'sdxl', 'main')];
const modelSnapshot = { models, status: 'loaded' };
const RECALLED: PromptHistoryItem = {
  negativePrompt: 'blurry, watermark',
  positivePrompt: 'a lighthouse at dusk',
} as PromptHistoryItem;
const noop = () => undefined;
const generationAdapter = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: noop },
} as unknown as GenerationUiAdapter;

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;

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
vi.mock('@features/models/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ModelSelect: () => null,
}));
vi.mock('@features/gallery/mediaSlot', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryMediaSlot: () => null,
}));
// The prompt field's own history list is Generation's; this stands in for choosing an entry from it.
vi.mock('@features/generation/components', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  NegativePromptField: () => null,
  PositivePromptField: ({ onUsePrompt }: { onUsePrompt: (prompt: PromptHistoryItem) => void }) => (
    <button
      aria-label="Use history entry"
      data-testid="use-history-entry"
      type="button"
      onClick={() => onUsePrompt(RECALLED)}
    />
  ),
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
  useOptionalWorkbenchCommands: () => store.commands,
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Widget = 'upscale' | 'video';

/** Upscale shares Generate's prompt draft; Video keeps its own prompts. */
const promptOwner = (widget: Widget) => (widget === 'upscale' ? 'generate' : 'video');

const readPrompts = (widget: Widget) => {
  const values = getProjectWidgetValues(store.getSnapshot().activeProject, promptOwner(widget));

  return { negativePrompt: values.negativePrompt, positivePrompt: values.positivePrompt };
};

const render = async (widget: Widget) => {
  const defaults =
    widget === 'video' ? createDefaultVideoWidgetValues(models) : createDefaultUpscaleWidgetValues(models);
  store.commands.widgets.patchValues(widget, { ...defaults });
  store.commands.widgets.patchValues(promptOwner(widget), {
    negativePrompt: 'low quality',
    negativePromptEnabled: true,
    positivePrompt: 'a harbour at noon',
  });

  await act(() =>
    root.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={generationAdapter}>
            <VideoUiAdapterProvider>
              <UpscaleUiAdapterProvider>
                {widget === 'video' ? <VideoWidgetView /> : <UpscaleWidgetView />}
              </UpscaleUiAdapterProvider>
            </VideoUiAdapterProvider>
          </GenerationUiProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );

  if (widget === 'upscale') {
    // The prompts live in the collapsed detail guidance section.
    await act(() => {
      [...host.querySelectorAll<HTMLButtonElement>('button')]
        .find((button) => button.textContent?.includes('widgets.upscale.detailGuidance'))
        ?.click();
    });
  }
};

const useHistoryEntry = () =>
  act(() => host.querySelector<HTMLButtonElement>('[data-testid="use-history-entry"]')?.click());

/** The Undo the recall's notice offers, pressed as its toast button would. */
const pressUndo = () => {
  const undo = store
    .getSnapshot()
    .notifications.flatMap((notification) => notification.actions ?? [])
    .find((action) => action.kind === 'undo');

  expect(undo).toBeDefined();
  act(() => runNotificationUndo((undo as { undoId: string }).undoId));
};

beforeEach(() => {
  store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('prompt history recall in Upscale and Video', () => {
  it.each(['upscale', 'video'] as const)('%s: Undo puts back the prompts a recall replaced', async (widget) => {
    await render(widget);
    await useHistoryEntry();

    expect(readPrompts(widget)).toEqual({
      negativePrompt: 'blurry, watermark',
      positivePrompt: 'a lighthouse at dusk',
    });
    expect(store.getSnapshot().notifications[0]?.title).toBe('widgets.generate.promptHistoryUsed');

    pressUndo();

    expect(readPrompts(widget)).toEqual({ negativePrompt: 'low quality', positivePrompt: 'a harbour at noon' });
  });

  it.each(['upscale', 'video'] as const)('%s: Undo keeps an edit made since the recall', async (widget) => {
    await render(widget);
    await useHistoryEntry();
    act(() => store.commands.widgets.patchValues(promptOwner(widget), { negativePrompt: 'typed afterwards' }));

    pressUndo();

    expect(readPrompts(widget)).toEqual({ negativePrompt: 'typed afterwards', positivePrompt: 'a harbour at noon' });
  });
});
