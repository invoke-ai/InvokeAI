/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { NegativePromptField } from '@features/generation/ui/promptFields/NegativePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const MODEL_CATALOG = [{ base: 'sdxl', name: 'easynegative', type: 'embedding' }];
const SELECTED_MODEL = { base: 'sdxl', name: 'Juggernaut', trigger_phrases: ['jugg'] };

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ models: { catalog: MODEL_CATALOG, ensureLoaded: vi.fn() } }),
}));

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({
    queryFn: () => Promise.resolve([{ id: 'wildcard-1', name: 'colors', values: ['red'] }]),
    queryKey: ['generation', 'wildcards'],
  }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (templateNegativePrompt: string | null, isTemplateViewMode = true) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <NegativePromptField
            heightPx={56}
            isEnabled
            isTemplateViewMode={isTemplateViewMode}
            loras={[]}
            projectId="project-1"
            selectedModel={SELECTED_MODEL as never}
            showSyntaxHighlighting={false}
            templateNegativePrompt={templateNegativePrompt}
            value="blurry"
            onChange={vi.fn()}
            onEnabledChange={vi.fn()}
            onResizeEnd={vi.fn()}
            onTemplateViewModeChange={vi.fn()}
          />
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  return host.querySelector('textarea')!;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the negative prompt in template view mode', () => {
  it('shows the merged text when the template has a negative side', async () => {
    const textarea = await render('lowres, {prompt}');

    expect(textarea.readOnly).toBe(true);
    expect(textarea.value).toBe('lowres, blurry');
  });

  // Most templates carry no negative side at all. Treating that empty string as
  // something to view made this field read-only for no reason, showing the
  // authored text with the merge's trailing space hanging off the end of it.
  it('stays editable when the template has none', async () => {
    const textarea = await render('');

    expect(textarea.readOnly).toBe(false);
    expect(textarea.value).toBe('blurry');
  });

  it('stays editable when no template is applied', async () => {
    const textarea = await render(null);

    expect(textarea.readOnly).toBe(false);
    expect(textarea.value).toBe('blurry');
    expect(host!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.addPromptTrigger"]')?.disabled).toBe(
      false
    );
  });
});

it('offers phrases and embeddings in the full picker but excludes wildcards', async () => {
  await render(null, false);

  await act(async () => {
    await userEvent.click(host!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.addPromptTrigger"]')!);
  });
  await vi.waitFor(() => expect(document.body.textContent).toContain('easynegative'));

  expect(document.body.textContent).toContain('jugg');
  expect(document.body.textContent).not.toContain('colors');
});
