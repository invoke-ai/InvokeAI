import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
import {
  captureRowInteractionStyles,
  expectRowInteractionStylesToMatch,
} from '@features/generation/ui/promptFields/promptFieldsBrowserTestUtils';
import { PromptTextarea } from '@features/generation/ui/promptFields/PromptTextarea';
import { Row } from '@platform/ui/Row';
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

const config: DynamicPromptsConfig & { onChange: () => void } = {
  combinatorial: true,
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
          <Box aria-hidden bg="bg.emphasized" data-testid="row-hover-style-probe" />
          <Box aria-hidden bg="bg.muted" data-testid="popover-surface-style-probe" />
          <Row asChild>
            <button aria-label="Row probe" type="button">
              Row probe
            </button>
          </Row>
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
  it('shows expanded prompts on the standard popover surface', async () => {
    const { onUsePrompt } = await render('a {red|green} cat');

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));

    await act(async () => {
      await userEvent.click(findButton());
    });

    const rows = [...document.querySelectorAll('button')].filter((button) => button.textContent?.includes('a red cat'));
    const content = document.querySelector<HTMLElement>('[data-scope="popover"][data-part="content"]')!;
    const surface = getComputedStyle(
      host!.querySelector('[data-testid="popover-surface-style-probe"]')!
    ).backgroundColor;

    expect(rows.length).toBe(1);
    expect(document.body.textContent).toContain('a green cat');
    expect(getComputedStyle(content).backgroundColor).toBe(surface);

    await act(async () => {
      await userEvent.click(rows[0]!);
    });

    expect(onUsePrompt).toHaveBeenCalledWith('a red cat');
  });

  it('uses the shared Row interaction contract for each selectable expanded prompt', async () => {
    await render('a {red|green} cat');
    const probe = host!.querySelector<HTMLButtonElement>('button[aria-label="Row probe"]')!;
    const hoverBackgroundColor = getComputedStyle(
      host!.querySelector('[data-testid="row-hover-style-probe"]')!
    ).backgroundColor;
    const expected = await captureRowInteractionStyles(probe, hoverBackgroundColor);

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));
    await act(async () => {
      await userEvent.click(findButton());
    });

    const row = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) =>
        button.title === 'widgets.generate.dynamicPrompts.usePrompt' && button.textContent?.includes('a red cat')
    )!;

    await expectRowInteractionStylesToMatch(expected, row, hoverBackgroundColor);
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
