import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { ChakraProvider } from '@chakra-ui/react';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', () => ({ parseDynamicPrompts }));

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

const config: DynamicPromptsFieldConfig = {
  combinatorial: true,
  isSeedFixed: false,
  maxPrompts: 100,
  onChange: vi.fn(),
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

const render = async (prompt: string, onUsePrompt = vi.fn()) => {
  host = document.createElement('div');
  host.style.width = '400px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <PromptTextarea
            aria-label="Prompt"
            defaultHeightPx={100}
            highlightDynamicPrompts
            minHeightPx={60}
            readOnly
            resizeHandleAriaLabel="Resize prompt"
            showSyntaxHighlighting
            value={prompt}
          />
          <DynamicPromptsButton
            batchCount={2}
            config={config}
            positivePrompt={prompt}
            showSyntaxHighlighting
            onInsertText={vi.fn()}
            onUsePrompt={onUsePrompt}
          />
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  return { onUsePrompt };
};

const findButton = () => [...host!.querySelectorAll('button')].at(-1)!;

beforeEach(() => {
  parseDynamicPrompts.mockReset();
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a red cat', 'a green cat'] });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('dynamic prompts in the positive prompt field', () => {
  it('shows every expanded prompt and uses the one that was clicked', async () => {
    const { onUsePrompt } = await render('a {red|green} cat');

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));

    await act(async () => {
      await userEvent.click(findButton());
    });

    const rows = [...document.querySelectorAll('button')].filter((button) => button.textContent?.includes('a red cat'));

    expect(rows.length).toBe(1);
    expect(document.body.textContent).toContain('a green cat');

    await act(async () => {
      await userEvent.click(rows[0]!);
    });

    expect(onUsePrompt).toHaveBeenCalledWith('a red cat');
  });

  it('is a labeled primary at rest, and widens only to carry a count', async () => {
    await render('a plain cat');
    const plain = findButton().getBoundingClientRect();

    expect(findButton().textContent).toContain('widgets.generate.dynamicButton');
    expect(findButton().textContent).not.toMatch(/\d/);

    await act(() => root?.unmount());
    host?.remove();
    await render('a {red|green} cat');
    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));

    expect(findButton().getBoundingClientRect().width).toBeGreaterThan(plain.width);
  });

  it('stays inert and never expands a prompt with no dynamic syntax', async () => {
    await render('a plain cat');

    await act(async () => {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 700);
      });
    });

    expect(parseDynamicPrompts).not.toHaveBeenCalled();
    expect(findButton().textContent).not.toMatch(/\d/);
  });
});
