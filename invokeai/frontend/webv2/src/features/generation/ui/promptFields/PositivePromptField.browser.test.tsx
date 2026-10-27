/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { GenerateModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { parseAnimaPrompt } from '@features/generation/core/prompt/ast';
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

const renderPromptModel = (selectedModel: GenerateModelConfig, value = 'a cat+') =>
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
                value={value}
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

it('colours and weights attention syntax only while the selected model reads it', async () => {
  const sdxl: GenerateModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
  const anima: GenerateModelConfig = { base: 'anima', key: 'anima', name: 'Anima', type: 'main' };
  const krea: GenerateModelConfig = { base: 'krea-2', key: 'krea', name: 'Krea-2 Turbo', type: 'main' };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const render = renderPromptModel;
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

  await render(anima);
  expect(textarea().getAttribute('data-prompt-syntax-profile')).toBe('anima');
  expect(colouredSegments()).toEqual(['a cat', '++']);
  expect(await weightCat()).toBe(true);
  await expect.poll(() => textarea().value).toBe('a cat+++');
});

it.each([
  ['((red, blue).and() cat)2', 'red', 2.1, 'blue', 2],
  ['(cat)2next', 'next', 1.1, 'cat', 2],
  ['foo\\(bar\\)+ cat', 'cat', 1.1, 'foo(bar)', 1.1],
  ['.and(dog+', 'dog', 1.21, '.and(', 1],
] as const)(
  'keeps Anima attention semantics through the browser hotkey: %s',
  async (prompt, target, expected, other, otherWeight) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await renderPromptModel({ base: 'anima', key: 'anima', name: 'Anima', type: 'main' }, prompt);
    const input = host.querySelector('textarea')!;
    const caret = prompt.indexOf(target) + (target === 'next' ? 0 : 1);
    input.focus();
    input.setSelectionRange(caret, caret);
    await act(() => {
      expect(adjustFocusedPromptAttention('increment', false)).toBe(true);
    });
    await expect.poll(() => input.value).not.toBe(prompt);
    const parsed = parseAnimaPrompt(input.value);
    expect(parsed.text).toBe(parseAnimaPrompt(prompt).text);
    for (const [word, weight] of [
      [target, expected],
      [other, otherWeight],
    ] as const) {
      const index = parsed.text.indexOf(word);
      expect(parsed.ranges.find(({ start, end }) => start <= index && end > index)?.weight).toBeCloseTo(weight, 10);
    }
    expect(host.querySelector('pre')?.textContent).toContain(input.value);
  }
);
