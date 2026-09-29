/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ExpandPromptSuggestion } from '@features/generation/core/types';
import type { SavedPromptModels } from '@features/generation/ui/promptFields/PositivePromptActions';
import type { ChangeEvent } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { expandPrompt, imageToPrompt } from '@features/generation/data/promptUtilities';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: { translation: { widgets: { generate: { expandSuggestedModelMissing: '{{model}} is not installed' } } } },
  },
});

const TEXT_ONLY = { base: 'any', key: 'qwen', name: 'Qwen', source: 'Qwen/Qwen2.5-1.5B-Instruct', type: 'text_llm' };
const ENHANCER = {
  base: 'any',
  key: 'e2b',
  name: 'Gemma-4 E2B',
  source: 'google/gemma-4-E2B-it',
  supports_images: true,
  type: 'text_llm',
};

const prompt = (id: string, content: string) => ({
  content,
  id,
  isPublic: true,
  maxTokens: null,
  name: id,
  userId: 'system',
});
const VISION_A = { base: 'any', key: 'llava-a', name: 'LLaVA A', type: 'llava_onevision' };
const VISION_B = { base: 'any', key: 'llava-b', name: 'LLaVA B', type: 'llava_onevision' };

const PROMPTS = [prompt('default', 'DEFAULT'), prompt('t2v', 'T2V'), prompt('i2v', 'I2V')];

const FRAME = { height: 512, image_name: 'frame.png', width: 768 };
const LTX_SUGGESTION: ExpandPromptSuggestion = {
  image: FRAME,
  imageSystemPromptId: 'i2v',
  modelName: 'LTX-2.5 Prompt Enhancer (Gemma-4 E2B)',
  modelSource: 'google/gemma-4-E2B-it',
  systemPromptId: 't2v',
};

// Read at render time by the mocks below, so each test can set them before rendering.
let catalog: (typeof TEXT_ONLY | typeof ENHANCER | typeof VISION_A)[] = [];
let arePromptsLoading = false;

const StubSelect = ({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  options: { key: string; name: string }[];
  value: string | null;
}) => (
  <select
    aria-label={label}
    value={value ?? ''}
    onChange={(event: ChangeEvent<HTMLSelectElement>) => onChange(event.currentTarget.value)}
  >
    <option value="">none</option>
    {options.map((option) => (
      <option key={option.key} value={option.key}>
        {option.name}
      </option>
    ))}
  </select>
);

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  GenerationModelSelect: ({
    onChange,
    value,
  }: {
    onChange: (model: (typeof catalog)[number] | null) => void;
    value: string | null;
  }) => (
    <StubSelect
      label="text llm"
      options={catalog}
      value={value}
      onChange={(key) => onChange(catalog.find((model) => model.key === key) ?? null)}
    />
  ),
  useGenerationUi: () => ({
    account: { isAdmin: false, userId: 'system' },
    capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
    gallery: { selectedImage: { imageName: 'selected.png' } },
    models: { catalog, ensureLoaded: vi.fn(), openManager: vi.fn() },
    notifications: { reportError: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [] },
  }),
}));

vi.mock('@features/generation/ui/promptFields/SystemPromptsField', () => ({
  SystemPromptsField: ({
    onSelect,
    selectedId,
  }: {
    onSelect: (id: string | null) => void;
    selectedId: string | null;
  }) => (
    <StubSelect
      label="system prompt"
      options={PROMPTS.map(({ id, name }) => ({ key: id, name }))}
      value={selectedId}
      onChange={(id) => onSelect(id || null)}
    />
  ),
}));

vi.mock('@features/generation/ui/promptFields/useSystemPrompts', () => ({
  useSystemPrompts: () => ({
    isLoaded: !arePromptsLoading,
    isLoading: arePromptsLoading,
    prompts: arePromptsLoading ? [] : PROMPTS,
  }),
}));

vi.mock('@features/generation/data/promptUtilities', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  expandPrompt: vi.fn(() => Promise.resolve({ expanded_prompt: 'rewritten', seed: 1 })),
  imageToPrompt: vi.fn(() => Promise.resolve({ prompt: 'described' })),
}));

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.mocked(expandPrompt).mockClear();
  vi.mocked(imageToPrompt).mockClear();
  catalog = [TEXT_ONLY, ENHANCER];
  arePromptsLoading = false;
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderField = (suggestion: ExpandPromptSuggestion | null, savedPromptModels?: SavedPromptModels) =>
  root?.render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <DndContext>
            <PositivePromptField
              expandPromptSuggestion={suggestion}
              heightPx={96}
              loras={[]}
              projectId="project-1"
              savedPromptModels={savedPromptModels}
              selectedModel={undefined}
              showSyntaxHighlighting={false}
              value="she waves"
              onChange={vi.fn()}
              onResizeEnd={vi.fn()}
              onUsePrompt={vi.fn()}
            />
          </DndContext>
        </ChakraProvider>
      </QueryClientProvider>
    </I18nextProvider>
  );

const renderAndOpen = async (
  suggestion: ExpandPromptSuggestion | null,
  savedPromptModels?: SavedPromptModels,
  trigger = 'widgets.generate.expandPrompt'
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => renderField(suggestion, savedPromptModels));
  await act(() => host?.querySelector<HTMLButtonElement>(`[aria-label="${trigger}"]`)?.click());
};

const savedModels = (overrides: Partial<SavedPromptModels> = {}): SavedPromptModels => ({
  expandPromptModelKey: null,
  imageToPromptModelKey: null,
  onChange: vi.fn(),
  ...overrides,
});

const select = (label: string) => document.querySelector<HTMLSelectElement>(`select[aria-label="${label}"]`);
const choose = (label: string, value: string) =>
  act(() => {
    const element = select(label);
    if (element) {
      element.value = value;
      element.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
const expandButton = () =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find(
    (button) => button.textContent === 'widgets.generate.expand'
  );
const firstFrameCheckbox = () =>
  [...document.querySelectorAll('label')].find((label) =>
    label.textContent?.includes('widgets.generate.expandFromFirstFrame')
  );

const runExpand = async () => {
  await act(() => expandButton()?.click());
  expect(expandPrompt).toHaveBeenCalledOnce();
  return vi.mocked(expandPrompt).mock.calls[0][0];
};

it('starts from the suggested enhancer and describes from the first frame with the image prompt', async () => {
  await renderAndOpen(LTX_SUGGESTION);

  expect(select('text llm')?.value).toBe('e2b');
  expect(await runExpand()).toMatchObject({ image_name: 'frame.png', model_key: 'e2b', system_prompt: 'I2V' });
});

it('switches to the text prompt when the user leaves the frame out', async () => {
  await renderAndOpen(LTX_SUGGESTION);

  await act(() => firstFrameCheckbox()?.click());
  const request = await runExpand();
  expect(request.image_name).toBeUndefined();
  expect(request.system_prompt).toBe('T2V');
});

it('offers a different frame again after one was left out', async () => {
  await renderAndOpen(LTX_SUGGESTION);

  await act(() => firstFrameCheckbox()?.click());
  await act(() => renderField({ ...LTX_SUGGESTION, image: { ...FRAME, image_name: 'other.png' } }));
  expect(await runExpand()).toMatchObject({ image_name: 'other.png', system_prompt: 'I2V' });
});

it('uses the text prompt and says why for a model that cannot read images', async () => {
  await renderAndOpen(LTX_SUGGESTION);

  await choose('text llm', 'qwen');
  expect(firstFrameCheckbox()).toBeUndefined();
  expect(document.body.textContent).toContain('widgets.generate.expandFirstFrameUnreadable');
  const request = await runExpand();
  expect(request).toMatchObject({ model_key: 'qwen', system_prompt: 'T2V' });
  expect(request.image_name).toBeUndefined();
});

it("keeps the user's own system prompt over the suggestion", async () => {
  await renderAndOpen(LTX_SUGGESTION);

  await choose('system prompt', 'default');
  expect(await runExpand()).toMatchObject({ image_name: 'frame.png', system_prompt: 'DEFAULT' });
});

it('points to the starter models and falls back to an installed model when the enhancer is missing', async () => {
  catalog = [TEXT_ONLY];
  await renderAndOpen(LTX_SUGGESTION);

  expect(select('text llm')?.value).toBe('qwen');
  expect(document.body.textContent).toContain('LTX-2.5 Prompt Enhancer (Gemma-4 E2B) is not installed');
  expect(document.body.textContent).toContain('widgets.generate.expandFirstFrameUnreadable');
  expect(await runExpand()).toMatchObject({ model_key: 'qwen', system_prompt: 'T2V' });
});

it('waits for the system prompts before expanding', async () => {
  arePromptsLoading = true;
  await renderAndOpen(LTX_SUGGESTION);

  expect(expandButton()?.disabled).toBe(true);
});

it('starts from the first listed model without a suggestion', async () => {
  await renderAndOpen(null);

  // Listed by name: "Gemma-4 E2B" before "Qwen".
  expect(select('text llm')?.value).toBe('e2b');
  expect(document.body.textContent).not.toContain('is not installed');
  expect(await runExpand()).toMatchObject({ model_key: 'e2b' });
});

it("uses the surface's saved pick over the suggestion and saves a different one", async () => {
  const saved = savedModels({ expandPromptModelKey: 'qwen' });
  await renderAndOpen(LTX_SUGGESTION, saved);

  expect(select('text llm')?.value).toBe('qwen');
  await choose('text llm', 'e2b');
  expect(saved.onChange).toHaveBeenCalledExactlyOnceWith({ expandPromptModelKey: 'e2b' });

  // Once saved, picking the same model again saves nothing.
  await act(() => renderField(LTX_SUGGESTION, { ...saved, expandPromptModelKey: 'e2b' }));
  await choose('text llm', 'e2b');
  expect(saved.onChange).toHaveBeenCalledOnce();
});

it('falls back when the saved pick is no longer installed', async () => {
  await renderAndOpen(null, savedModels({ expandPromptModelKey: 'uninstalled' }));

  expect(select('text llm')?.value).toBe('e2b');
});

it('starts Image to Prompt from the first listed vision model and saves a different pick', async () => {
  catalog = [VISION_B, VISION_A];
  const saved = savedModels();
  await renderAndOpen(null, saved, 'widgets.generate.imageToPrompt');

  expect(select('text llm')?.value).toBe('llava-a');
  const generateButton = () =>
    [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'widgets.generate.generatePrompt'
    );
  await act(() => generateButton()?.click());
  expect(imageToPrompt).toHaveBeenCalledWith(
    expect.objectContaining({ image_name: 'selected.png', model_key: 'llava-a' })
  );

  await act(() => host?.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.imageToPrompt"]')?.click());
  await choose('text llm', 'llava-b');
  expect(saved.onChange).toHaveBeenCalledExactlyOnceWith({ imageToPromptModelKey: 'llava-b' });
});
