/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerateModelConfig } from '@features/generation/core/types';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { adjustFocusedPromptAttention } from '@features/generation/ui/promptFields/promptAttentionHotkeys';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  parseDynamicPrompts,
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
    gallery: { selectedImage: null },
    models: { catalog: [], ensureLoaded: vi.fn() },
    notifications: { reportError: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [] },
  }),
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
  isSeedFixed: false,
  maxPrompts: 100,
  onChange: vi.fn(),
  resample: true,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};
const PHOTO_TEMPLATE = { id: 'photo', name: 'Photo', negativePrompt: '', positivePrompt: 'photo of {prompt}' };

it('shows a JSON prompt under a template whole, as it generates, and never expands its braces', async () => {
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a red cup', 'a green cup'] });
  const layout =
    '{"bbox_coord_space":"abs_centralized","bbox_format":"yxyx","summary":{"caption":"a {red|green} cup"},"items":[]}';
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
                batchCount={2}
                dynamicPrompts={DYNAMIC_PROMPTS}
                heightPx={96}
                isTemplateViewMode
                loras={[]}
                projectId="project-1"
                promptTemplate={PHOTO_TEMPLATE}
                selectedModel={undefined}
                showSyntaxHighlighting
                value={layout}
                onChange={vi.fn()}
                onResizeEnd={vi.fn()}
                onTemplateViewModeChange={vi.fn()}
                onUsePrompt={vi.fn()}
              />
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    );
  });
  // Past the dynamic prompts debounce, when a dynamic prompt would have been parsed.
  await act(async () => {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 700);
    });
  });

  expect(host.querySelector('textarea')?.value).toBe(layout);
  // The merged-view overlay lines up with the textarea: no template text around the layout.
  expect(host.querySelector('pre')?.textContent).toBe(layout);
  const dynamicButton = [...host.querySelectorAll('button')].find((button) =>
    button.textContent?.includes('widgets.generate.dynamicButton')
  );
  expect(dynamicButton).toBeDefined();
  expect(dynamicButton?.textContent).not.toMatch(/\d/);
  expect(parseDynamicPrompts).not.toHaveBeenCalled();
});
