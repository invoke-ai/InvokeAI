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
import { commands, userEvent } from 'vitest/browser';

declare module 'vitest/browser' {
  interface BrowserCommands {
    imeCompose: (steps: readonly { kind: 'commit' | 'compose'; text: string }[]) => Promise<void>;
  }
}

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          generate: {
            expandSuggestedModelAvailable: 'Recommended: {{model}}',
            expandSuggestedModelMissing: '{{model}} is not installed',
            expandUseSuggestedModelLabel: 'Use {{model}}',
          },
        },
      },
    },
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

const MOSS_ENHANCER = { ...ENHANCER, key: 'e4b', name: 'Gemma 4 E4B', source: 'google/gemma-4-E4B-it' };
const MOSS_PROMPT_ID = '0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0011';
const MOSS_SUGGESTION: ExpandPromptSuggestion = {
  image: null,
  imageSystemPromptId: null,
  modelName: 'Gemma 4 E4B (Prompt Enhancer)',
  modelSource: 'google/gemma-4-E4B-it',
  systemPromptId: MOSS_PROMPT_ID,
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

const PROMPTS = [
  prompt('default', 'DEFAULT'),
  prompt('t2v', 'T2V'),
  prompt('i2v', 'I2V'),
  prompt(MOSS_PROMPT_ID, 'MOSS'),
];

const FRAME = { height: 512, image_name: 'frame.png', width: 768 };
const LTX_SUGGESTION: ExpandPromptSuggestion = {
  image: FRAME,
  imageSystemPromptId: 'i2v',
  modelName: 'LTX-2.5 Prompt Enhancer (Gemma-4 E2B)',
  modelSource: 'google/gemma-4-E2B-it',
  systemPromptId: 't2v',
};

const notices = vi.hoisted(() => ({ info: vi.fn(), offer: vi.fn(), undoable: vi.fn() }));

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
    notifications: { info: notices.info, offer: notices.offer, reportError: vi.fn(), undoable: notices.undoable },
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

beforeEach(() => {
  notices.info.mockClear();
  notices.offer.mockClear();
  notices.undoable.mockClear();
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

/** What the field was last rendered with, so a saved slot can render its own change back. */
let rendered: {
  onChange: (value: string) => void;
  projectId: string;
  savedPromptModels: SavedPromptModels | undefined;
  suggestion: ExpandPromptSuggestion | null;
} | null = null;

const renderField = (
  suggestion: ExpandPromptSuggestion | null,
  savedPromptModels?: SavedPromptModels,
  onChange: (value: string) => void = vi.fn(),
  projectId = 'project-1'
) => {
  rendered = { onChange, projectId, savedPromptModels, suggestion };
  root?.render(
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <DndContext>
            <PositivePromptField
              expandPromptSuggestion={suggestion}
              heightPx={96}
              loras={[]}
              projectId={projectId}
              savedPromptModels={savedPromptModels}
              selectedModel={undefined}
              showSyntaxHighlighting={false}
              value="she waves"
              onChange={onChange}
              onResizeEnd={vi.fn()}
              onUsePrompt={vi.fn()}
            />
          </DndContext>
        </ChakraProvider>
      </QueryClientProvider>
    </I18nextProvider>
  );
};

const renderAndOpen = async (
  suggestion: ExpandPromptSuggestion | null,
  savedPromptModels?: SavedPromptModels,
  trigger = 'widgets.generate.expandPrompt',
  onChange?: (value: string) => void
) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => renderField(suggestion, savedPromptModels, onChange));
  await act(() => host?.querySelector<HTMLButtonElement>(`[aria-label="${trigger}"]`)?.click());
};

/** A saved slot that, like Generate's draft, holds a saved pick by the next render. */
const savedModels = (overrides: Partial<SavedPromptModels> = {}): SavedPromptModels => {
  const onChange = vi.fn((patch: Partial<SavedPromptModels>) => {
    if (rendered?.savedPromptModels?.onChange === onChange) {
      renderField(
        rendered.suggestion,
        { ...rendered.savedPromptModels, ...patch },
        rendered.onChange,
        rendered.projectId
      );
    }
  });
  return { expandPromptModelKey: null, imageToPromptModelKey: null, ...overrides, onChange };
};

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

const useSuggestedButton = (name: string) =>
  document.querySelector<HTMLButtonElement>(`button[aria-label="Use ${name}"]`);

it("keeps the surface's saved pick over an installed suggestion, also after a remount, and saves a different one", async () => {
  const saved = savedModels({ expandPromptModelKey: 'qwen' });
  await renderAndOpen(LTX_SUGGESTION, saved);

  expect(select('text llm')?.value).toBe('qwen');
  expect(document.body.textContent).toContain('Recommended: Gemma-4 E2B');
  await act(() => root?.unmount());
  host?.remove();
  await renderAndOpen(LTX_SUGGESTION, saved);
  expect(select('text llm')?.value).toBe('qwen');

  await choose('text llm', 'e2b');
  expect(saved.onChange).toHaveBeenCalledExactlyOnceWith({ expandPromptModelKey: 'e2b' });
});

it('switches to the recommended MOSS-Image enhancer from the hint and saves it', async () => {
  catalog = [TEXT_ONLY, ENHANCER, MOSS_ENHANCER];
  const saved = savedModels({ expandPromptModelKey: 'e2b' });
  await renderAndOpen(MOSS_SUGGESTION, saved);

  expect(select('text llm')?.value).toBe('e2b');
  await act(() => useSuggestedButton('Gemma 4 E4B')?.click());
  expect(select('text llm')?.value).toBe('e4b');
  expect(useSuggestedButton('Gemma 4 E4B')).toBeNull();
  expect(saved.onChange).toHaveBeenCalledExactlyOnceWith({ expandPromptModelKey: 'e4b' });
  expect(await runExpand()).toMatchObject({ model_key: 'e4b', system_prompt: 'MOSS' });
});

it('preselects an installed suggestion when nothing is saved, without the hint', async () => {
  catalog = [TEXT_ONLY, ENHANCER, MOSS_ENHANCER];
  await renderAndOpen(MOSS_SUGGESTION, savedModels());

  expect(select('text llm')?.value).toBe('e4b');
  expect(useSuggestedButton('Gemma 4 E4B')).toBeNull();
});

it('recognises the MOSS-Image enhancer installed in place from a local folder', async () => {
  catalog = [TEXT_ONLY, { ...MOSS_ENHANCER, source: 'E:/ModelStuffSSD/gemma-4-E4B-it' }];
  await renderAndOpen(MOSS_SUGGESTION, savedModels());

  expect(select('text llm')?.value).toBe('e4b');
  expect(document.body.textContent).not.toContain('is not installed');
});

it('points to the MOSS-Image enhancer starter model when it is not installed', async () => {
  await renderAndOpen(MOSS_SUGGESTION, savedModels({ expandPromptModelKey: 'e2b' }));

  expect(select('text llm')?.value).toBe('e2b');
  expect(document.body.textContent).toContain('Gemma 4 E4B (Prompt Enhancer) is not installed');
  expect(document.body.textContent).not.toContain('Recommended:');
});

it("starts another project from its own saved Expand Prompt model, not the previous project's pick", async () => {
  await renderAndOpen(null, savedModels());
  await choose('text llm', 'qwen');
  expect(select('text llm')?.value).toBe('qwen');

  await act(() => renderField(null, savedModels({ expandPromptModelKey: 'e2b' }), undefined, 'project-2'));
  expect(select('text llm')?.value).toBe('e2b');
});

it('keeps the session pick of a surface without a saved slot across a project switch', async () => {
  await renderAndOpen(null);
  await choose('text llm', 'qwen');

  await act(() => renderField(null, undefined, undefined, 'project-2'));
  expect(select('text llm')?.value).toBe('qwen');
});

it('falls back when the saved pick is no longer installed', async () => {
  await renderAndOpen(null, savedModels({ expandPromptModelKey: 'uninstalled' }));

  expect(select('text llm')?.value).toBe('e2b');
});

const isPopoverOpen = (title: string) =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="popover"][data-part="content"]')].some(
    (content) => content.dataset.state === 'open' && content.textContent?.includes(title)
  );
// A real click on the page outside the popover; zag settles the dismissal outside act.
const clickOutside = async () => {
  const outside = document.createElement('div');
  outside.style.height = '200px';
  document.body.append(outside);
  await userEvent.click(outside);
  outside.remove();
  await settle();
};
const settle = () =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 100);
    });
  });
/** A run that stays in flight until `finish` is called. */
const pendingRun = () => {
  let finish: (() => void) | undefined;
  const running = new Promise<void>((resolve) => {
    finish = resolve;
  });
  return {
    finish: () => {
      finish?.();
      return settle();
    },
    running,
  };
};

it('closes an idle Expand Prompt popover on an outside click', async () => {
  await renderAndOpen(null);
  expect(isPopoverOpen('widgets.generate.expand')).toBe(true);

  await clickOutside();
  expect(isPopoverOpen('widgets.generate.expand')).toBe(false);
});

const trigger = (label: string) => host?.querySelector<HTMLButtonElement>(`[aria-label="${label}"]`);
const isSpinning = (button: HTMLButtonElement | null | undefined) =>
  button?.querySelector('.chakra-spinner') !== null && button?.getAttribute('aria-busy') === 'true';

const promptBox = () =>
  host?.querySelector<HTMLTextAreaElement>('textarea[aria-label="widgets.generate.positivePrompt"]');
const buttonWithText = (text: string) =>
  [...document.querySelectorAll<HTMLButtonElement>('button')].find((button) => button.textContent === text);

/** The Undo the last landed result offered, as its toast would run it. */
const undoLastResult = () => {
  const [, notice] = notices.undoable.mock.calls.at(-1) as [string, { onUndo: () => void }];
  return act(() => notice.onUndo());
};

it('writes a result that lands while the popover is open as an edit, closes it, and offers an Undo', async () => {
  const run = pendingRun();
  const onChange = vi.fn();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null, undefined, 'widgets.generate.expandPrompt', onChange);

  await act(() => expandButton()?.click());
  expect(isSpinning(trigger('widgets.generate.expandPrompt'))).toBe(true);

  await run.finish();
  expect(isPopoverOpen('widgets.generate.expand')).toBe(false);
  expect(isSpinning(trigger('widgets.generate.expandPrompt'))).toBe(false);
  expect(promptBox()?.value).toBe('x');
  expect(onChange).toHaveBeenLastCalledWith('x');
  expect(document.activeElement).toBe(promptBox());
  expect(notices.undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.promptExpanded', {
    onUndo: expect.any(Function),
    projectId: 'project-1',
    supersedes: 'generate-prompt:project-1',
  });
  // What Ctrl+Z runs: the browser's own undo, which a controlled value write would have wiped.
  await act(() => {
    promptBox()?.focus();
    document.execCommand('undo');
  });
  expect(promptBox()?.value).toBe('she waves');
});

it('keeps running once the popover closes, and lands without taking focus from the field the user typed in', async () => {
  const run = pendingRun();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null);
  const elsewhere = document.createElement('textarea');
  document.body.append(elsewhere);

  await act(() => expandButton()?.click());
  await clickOutside();
  expect(isPopoverOpen('widgets.generate.expand')).toBe(false);
  expect(isSpinning(trigger('widgets.generate.expandPrompt'))).toBe(true);

  await userEvent.type(elsewhere, 'notes');
  await run.finish();

  expect(promptBox()?.value).toBe('x');
  expect(document.activeElement).toBe(elsewhere);
  expect(elsewhere.value).toBe('notes');
  await undoLastResult();
  expect(promptBox()?.value).toBe('she waves');
  elsewhere.remove();
});

it('leaves a composing IME in another field alone when a result lands', async () => {
  const run = pendingRun();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null);
  const elsewhere = document.createElement('textarea');
  document.body.append(elsewhere);

  await act(() => expandButton()?.click());
  await clickOutside();
  await act(() => elsewhere.focus());
  await commands.imeCompose([{ kind: 'compose', text: 'にほ' }]);
  await run.finish();
  await commands.imeCompose([{ kind: 'commit', text: '日本' }]);

  expect(promptBox()?.value).toBe('x');
  expect(document.activeElement).toBe(elsewhere);
  expect(elsewhere.value).toBe('日本');
  elsewhere.remove();
});

it('drops the result of a request the user stopped waiting for', async () => {
  const run = pendingRun();
  const onChange = vi.fn();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null, undefined, 'widgets.generate.expandPrompt', onChange);

  await act(() => expandButton()?.click());
  await act(() => buttonWithText('widgets.generate.stopWaiting')?.click());
  expect(isSpinning(trigger('widgets.generate.expandPrompt'))).toBe(false);

  await run.finish();
  expect(promptBox()?.value).toBe('she waves');
  expect(onChange).not.toHaveBeenCalled();
  expect(notices.undoable).not.toHaveBeenCalled();
});

it('offers the result in a toast instead of overwriting a prompt edited while it ran', async () => {
  const run = pendingRun();
  vi.mocked(expandPrompt).mockImplementationOnce(() =>
    run.running.then(() => ({ expanded_prompt: 'rewritten', seed: 1 }))
  );
  await renderAndOpen(null);

  await act(() => expandButton()?.click());
  await act(() => promptBox()?.setSelectionRange(9, 9));
  await userEvent.type(promptBox()!, ' hello');
  await run.finish();

  expect(promptBox()?.value).toBe('she waves hello');
  expect(notices.undoable).not.toHaveBeenCalled();
  const [title, offer] = notices.offer.mock.calls[0] as [
    string,
    { actions: { label: string; onClick: () => void }[]; message: string },
  ];
  expect(title).toBe('widgets.generate.promptChangedWhileRunning');
  expect(offer.message).toBe('rewritten');
  expect(offer.actions.map((action) => action.label)).toEqual(['widgets.generate.replacePrompt', 'common.discard']);

  await act(() => offer.actions[0]!.onClick());
  expect(promptBox()?.value).toBe('rewritten');
  await undoLastResult();
  expect(promptBox()?.value).toBe('she waves hello');
});

it('drops a result for a project this field no longer shows', async () => {
  const run = pendingRun();
  const onChange = vi.fn();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null, undefined, 'widgets.generate.expandPrompt', onChange);

  await act(() => expandButton()?.click());
  await act(() => renderField(null, undefined, onChange, 'project-2'));
  await run.finish();

  expect(promptBox()?.value).toBe('she waves');
  expect(onChange).not.toHaveBeenCalled();
  expect(notices.offer).not.toHaveBeenCalled();
  expect(notices.undoable).not.toHaveBeenCalled();
});

it('drops the request when its field unmounts, so a remounted field is never written by it', async () => {
  const run = pendingRun();
  const onChange = vi.fn();
  vi.mocked(expandPrompt).mockImplementationOnce(() => run.running.then(() => ({ expanded_prompt: 'x', seed: 1 })));
  await renderAndOpen(null, undefined, 'widgets.generate.expandPrompt', onChange);

  await act(() => expandButton()?.click());
  await act(() => root?.render(null));
  await act(() => renderField(null, undefined, onChange));
  await run.finish();

  expect(promptBox()?.value).toBe('she waves');
  expect(onChange).not.toHaveBeenCalled();
  expect(notices.undoable).not.toHaveBeenCalled();
});

// Shares Expand's request handling; this covers its own wiring.
it('runs Image to Prompt and writes its prompt', async () => {
  catalog = [VISION_A];
  const run = pendingRun();
  vi.mocked(imageToPrompt).mockImplementationOnce(() => run.running.then(() => ({ prompt: 'described' })));
  await renderAndOpen(null, undefined, 'widgets.generate.imageToPrompt');
  const generateButton = () => buttonWithText('widgets.generate.generatePrompt');

  await act(() => generateButton()?.click());
  expect(isSpinning(trigger('widgets.generate.imageToPrompt'))).toBe(true);

  await run.finish();
  expect(isSpinning(trigger('widgets.generate.imageToPrompt'))).toBe(false);
  expect(isPopoverOpen('widgets.generate.generatePrompt')).toBe(false);
  expect(promptBox()?.value).toBe('described');
  expect(notices.undoable).toHaveBeenCalledWith('widgets.generate.promptFromImage', expect.anything());
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

it("starts another project from its own saved Image to Prompt model, not the previous project's pick", async () => {
  catalog = [VISION_A, VISION_B];
  await renderAndOpen(null, savedModels(), 'widgets.generate.imageToPrompt');
  await choose('text llm', 'llava-b');
  expect(select('text llm')?.value).toBe('llava-b');

  await act(() => renderField(null, savedModels({ imageToPromptModelKey: 'llava-a' }), undefined, 'project-2'));
  expect(select('text llm')?.value).toBe('llava-a');
});
