/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { GenerateModelConfig } from '@features/generation/core/types';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { adjustFocusedPromptAttention } from '@features/generation/ui/promptFields/promptAttentionHotkeys';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

type HistoryItem = { negativePrompt: string | null; positivePrompt: string };

/** The project's history as the workbench keeps it: Clear and Restore change it and re-render its readers. */
const ui = vi.hoisted(() => {
  let history: readonly HistoryItem[] = [];
  const listeners = new Set<() => void>();
  const set = (next: readonly HistoryItem[]) => {
    history = next;
    listeners.forEach((listener) => listener());
  };

  return {
    clearHistory: (_projectId: string) => set([]),
    getHistory: () => history,
    restoreHistory: (items: readonly HistoryItem[], _projectId: string) => set([...history, ...items]),
    setHistory: set,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    undoable: vi.fn(),
  };
});
const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    promptHistory: {
      clear: ui.clearHistory,
      // oxlint-disable-next-line react-hooks/rules-of-hooks -- the port's history is a store the adapter reads.
      items: useSyncExternalStore(ui.subscribe, ui.getHistory),
      remove: vi.fn(),
      restore: ui.restoreHistory,
    },
    capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
    gallery: { selectedImage: null },
    models: { catalog: [], ensureLoaded: vi.fn() },
    notifications: { reportError: vi.fn(), undoable: ui.undoable },
    project: { activeProjectId: 'project-1' },
  }),
}));

vi.mock('@features/generation/data/promptUtilities', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  parseDynamicPrompts,
}));

vi.mock('@features/generation/data/wildcards', async (importOriginal) => ({
  // Partial: the real module's other exports (constants, mappers) stay available to what imports them.
  ...(await importOriginal<Record<string, unknown>>()),
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('keeps positive prompt actions enabled when stale view mode has no active template', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            {/* Match production image-drop ownership with DndContext. */}
            <DndContext>
              <PositivePromptField
                heightPx={96}
                isTemplateViewMode
                loras={[]}
                projectId="project-1"
                selectedModel={undefined}
                showSyntaxHighlighting={false}
                value="a cat"
                onChange={vi.fn()}
                onResizeEnd={vi.fn()}
                onUsePrompt={vi.fn()}
              />
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    );
  });

  expect(host.querySelector('textarea')?.readOnly).toBe(false);
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.addPromptTrigger"]')?.disabled).toBe(
    false
  );
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.expandPrompt"]')?.disabled).toBe(false);
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.imageToPrompt"]')?.disabled).toBe(false);
});

it('colours and weights attention syntax only while the selected model reads it', async () => {
  const sdxl: GenerateModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
  const krea: GenerateModelConfig = { base: 'krea-2', key: 'krea', name: 'Krea-2 Turbo', type: 'main' };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const render = (selectedModel: GenerateModelConfig) =>
    act(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={new QueryClient()}>
            <ChakraProvider value={system}>
              <DndContext>
                <PositivePromptField
                  heightPx={96}
                  loras={[]}
                  projectId="project-1"
                  selectedModel={selectedModel}
                  showSyntaxHighlighting
                  value="a cat+"
                  onChange={vi.fn()}
                  onResizeEnd={vi.fn()}
                  onUsePrompt={vi.fn()}
                />
              </DndContext>
            </ChakraProvider>
          </QueryClientProvider>
        </I18nextProvider>
      );
    });
  const textarea = () => host!.querySelector('textarea')!;
  const colouredSegments = () => [...host!.querySelectorAll('pre span')].map((span) => span.textContent);
  /** Select "cat" and run the weighting command the Ctrl+Up hotkey runs. */
  const weightCat = async () => {
    let weighted = false;

    textarea().focus();
    textarea().setSelectionRange(2, 5);
    await act(() => {
      weighted = adjustFocusedPromptAttention('increment', false);
    });

    return weighted;
  };

  await render(sdxl);
  expect(colouredSegments()).toEqual(['a cat', '+']);

  await render(krea);
  expect(host.querySelector('pre')).toBeNull();
  expect(await weightCat()).toBe(false);
  expect(textarea().value).toBe('a cat+');

  await render(sdxl);
  expect(colouredSegments()).toEqual(['a cat', '+']);
  expect(await weightCat()).toBe(true);
  await expect.poll(() => textarea().value).toBe('a cat++');
});

const DYNAMIC_PROMPTS: DynamicPromptsFieldConfig = {
  combinatorial: true,
  enabled: true,
  isSeedFixed: false,
  maxPrompts: 100,
  onChange: vi.fn(),
  resample: true,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

const PHOTO_TEMPLATE = { id: 'photo', name: 'Photo', negativePrompt: '', positivePrompt: 'photo of {prompt}' };

const renderField = async ({
  batchCount,
  dynamicPrompts = null,
  isTemplateViewMode,
  onChange = vi.fn(),
  onTemplateViewModeChange,
  onUsePrompt = vi.fn(),
  promptTemplate,
  showSyntaxHighlighting = false,
  value,
}: {
  batchCount?: number;
  dynamicPrompts?: DynamicPromptsFieldConfig | null;
  isTemplateViewMode?: boolean;
  onChange?: (value: string) => void;
  onTemplateViewModeChange?: (isTemplateViewMode: boolean) => void;
  onUsePrompt?: (prompt: { negativePrompt: string | null; positivePrompt: string }) => void;
  promptTemplate?: PromptTemplateSnapshot;
  showSyntaxHighlighting?: boolean;
  value: string;
}) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <DndContext>
              <PositivePromptField
                batchCount={batchCount}
                dynamicPrompts={dynamicPrompts}
                heightPx={96}
                isTemplateViewMode={isTemplateViewMode}
                loras={[]}
                projectId="project-1"
                promptTemplate={promptTemplate}
                selectedModel={undefined}
                showSyntaxHighlighting={showSyntaxHighlighting}
                value={value}
                onChange={onChange}
                onResizeEnd={vi.fn()}
                onTemplateViewModeChange={onTemplateViewModeChange}
                onUsePrompt={onUsePrompt}
              />
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    );
  });
};

const textarea = () => host!.querySelector('textarea')!;
const buttonNamed = (name: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.getAttribute('aria-label') === name || button.textContent === name
  )!;

describe('prompt history', () => {
  beforeEach(() => {
    ui.setHistory([{ negativePrompt: null, positivePrompt: 'a fox in snow' }]);
    ui.undoable.mockClear();
  });

  const listedPrompts = () =>
    [...document.querySelectorAll('[data-scope="popover"][data-part="content"] p')]
      .map((line) => line.textContent ?? '')
      .filter((text) => text.includes('a fox in snow'));

  it('uses an entry as an edit of the prompt, so the browser undo brings the old prompt back', async () => {
    const onUsePrompt = vi.fn();

    await renderField({ onUsePrompt, value: 'a cat' });
    await act(() => buttonNamed('widgets.generate.promptHistory').click());
    await act(() => buttonNamed('widgets.generate.usePrompt').click());

    expect(onUsePrompt).toHaveBeenCalledWith({ negativePrompt: null, positivePrompt: 'a fox in snow' });
    expect(textarea().value).toBe('a fox in snow');
    // The popover hands focus to the prompt, where Ctrl+Z undoes the write, not back to its trigger.
    await expect.poll(() => document.activeElement).toBe(textarea());

    await act(() => {
      textarea().focus();
      document.execCommand('undo');
    });
    expect(textarea().value).toBe('a cat');
  });

  it('clears without asking, with an Undo that puts every entry back in the same project', async () => {
    await renderField({ value: 'a cat' });
    await act(() => buttonNamed('widgets.generate.promptHistory').click());
    expect(listedPrompts()).toHaveLength(1);

    await act(() => buttonNamed('common.clear').click());

    expect(listedPrompts()).toHaveLength(0);
    expect(document.body.textContent).toContain('widgets.generate.noPromptHistoryYet');
    expect(ui.undoable).toHaveBeenCalledOnce();
    const [, notice] = ui.undoable.mock.calls[0] as [string, { onUndo: () => void; projectId: string }];
    expect(notice.projectId).toBe('project-1');

    await act(() => notice.onUndo());
    expect(listedPrompts()).toHaveLength(1);
  });
});

describe('the dynamic prompts notice under the prompt', () => {
  /** The notice's live region: mounted empty, so screen readers announce what appears in it. */
  const notice = () =>
    host!.querySelector('[aria-live="polite"][role="status"]:not([id^="DndLiveRegion"])')?.textContent;

  beforeEach(() => {
    parseDynamicPrompts.mockReset();
    parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['wall painted'] });
  });

  it('warns that text after # is dropped, and turns dynamic prompts off from there', async () => {
    const onChange = vi.fn();

    await renderField({ dynamicPrompts: { ...DYNAMIC_PROMPTS, onChange }, value: 'wall painted #ff0000' });

    expect(notice()).toContain('widgets.generate.dynamicPrompts.commentDropsText');

    await act(() => buttonNamed('widgets.generate.dynamicPrompts.turnOff').click());
    expect(onChange).toHaveBeenCalledWith({ enabled: false });
  });

  it('keeps # literal and asks the server nothing while dynamic prompts are off', async () => {
    await renderField({ dynamicPrompts: { ...DYNAMIC_PROMPTS, enabled: false }, value: 'wall painted #ff0000 {a|b}' });
    await act(async () => {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 700);
      });
    });

    expect(notice()).toBe('');
    expect(parseDynamicPrompts).not.toHaveBeenCalled();
  });

  it('reports a failed expansion with a retry and the way to run the prompt as typed', async () => {
    parseDynamicPrompts.mockRejectedValueOnce(new Error('offline'));

    await renderField({ dynamicPrompts: DYNAMIC_PROMPTS, value: 'a {red|green} cat' });
    await vi.waitFor(() => expect(notice()).toContain('widgets.generate.dynamicPrompts.expansionFailed'));
    expect(buttonNamed('widgets.generate.dynamicPrompts.turnOff')).toBeTruthy();

    await act(async () => {
      await userEvent.click(buttonNamed('common.retry'));
    });
    await vi.waitFor(() => expect(notice()).toBe(''));
    expect(parseDynamicPrompts).toHaveBeenCalledTimes(2);
  });
});

it('shows a JSON prompt under a template whole, as it generates, and never expands its braces', async () => {
  parseDynamicPrompts.mockReset();
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a red cup', 'a green cup'] });
  const layout =
    '{"bbox_coord_space":"abs_centralized","bbox_format":"yxyx","summary":{"caption":"a {red|green} cup"},"items":[]}';

  await renderField({
    batchCount: 2,
    dynamicPrompts: DYNAMIC_PROMPTS,
    isTemplateViewMode: true,
    promptTemplate: PHOTO_TEMPLATE,
    showSyntaxHighlighting: true,
    value: layout,
  });
  // Past the dynamic prompts debounce, when a dynamic prompt would have been parsed.
  await act(async () => {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 700);
    });
  });

  expect(textarea().value).toBe(layout);
  // The merged-view overlay lines up with the textarea: no template text around the layout.
  expect(host!.querySelector('pre')?.textContent).toBe(layout);
  const dynamicButton = [...host!.querySelectorAll('button')].find((button) =>
    button.textContent?.includes('widgets.generate.dynamicButton')
  );
  expect(dynamicButton).toBeDefined();
  expect(dynamicButton?.textContent).not.toMatch(/\d/);
  expect(parseDynamicPrompts).not.toHaveBeenCalled();
});
