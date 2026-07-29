import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { ChakraProvider } from '@chakra-ui/react';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
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

// The wildcards tab fetches the catalog on mount; the preview assertions do not need it.
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

// i18n is not bootstrapped in browser tests, so labels come back as keys; the
// trigger is the last button rendered inside the host.
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
  it('marks braces and separators in the highlight underlay', async () => {
    await render('a {red|green} cat');

    const underlay = host!.querySelector('[aria-hidden="true"] pre')!;
    const spans = [...underlay.querySelectorAll('span')].map((span) => span.textContent);

    // Each brace and separator is its own span, so they can be coloured apart
    // from the surrounding text.
    expect(spans).toContain('{');
    expect(spans).toContain('|');
    expect(spans).toContain('}');
  });

  it('shows the expanded count on the button and the prompts in the popover', async () => {
    const { onUsePrompt } = await render('a {red|green} cat');

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));

    await act(async () => {
      await userEvent.click(findButton());
    });

    const rows = [...document.querySelectorAll('button')].filter((button) => button.textContent?.includes('a red cat'));

    expect(rows.length).toBe(1);
    expect(document.body.textContent).toContain('a green cat');

    // Clicking a previewed prompt adopts that concrete expansion.
    await act(async () => {
      await userEvent.click(rows[0]!);
    });

    expect(onUsePrompt).toHaveBeenCalledWith('a red cat');
  });

  it('uses the shared Row interaction contract for each selectable expanded prompt', async () => {
    const { onUsePrompt } = await render('a {red|green} cat');
    const probe = host!.querySelector<HTMLButtonElement>('button[aria-label="Row probe"]')!;
    const expected = await getRowInteractionStyles(probe);

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));
    await act(async () => {
      await userEvent.click(findButton());
    });

    const row = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) =>
        button.title === 'widgets.generate.dynamicPrompts.usePrompt' && button.textContent?.includes('a red cat')
    )!;

    await expectRowInteractionsToMatch(expected, row);

    await act(async () => {
      await userEvent.click(row);
    });

    expect(onUsePrompt).toHaveBeenCalledWith('a red cat');
  });

  it('colours attention syntax inside the previewed prompts', async () => {
    parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a (red)1.2 cat', 'a green+ cat'] });
    await render('a {(red)1.2|green+} cat');

    await vi.waitFor(() => expect(findButton().textContent).toContain('2'));
    await act(async () => {
      await userEvent.click(findButton());
    });

    // An expanded prompt has no dynamic syntax left, so what is worth colouring
    // in the preview is the attention weighting that survived expansion.
    const row = [...document.querySelectorAll('button')].find((button) =>
      button.textContent?.includes('a (red)1.2 cat')
    )!;
    const weights = [...row.querySelectorAll('span')].map((span) => span.textContent);

    expect(weights).toContain('1.2');
    expect(weights).toContain('(');
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

const getRowInteractionStyles = async (probe: HTMLButtonElement) => {
  await act(async () => {
    await userEvent.tab();
    await userEvent.hover(probe);
    await waitForTransition();
  });
  const expected = getInteractionStyles(probe);

  await act(async () => {
    await userEvent.unhover(probe);
  });

  return expected;
};

const expectRowInteractionsToMatch = async (
  expected: ReturnType<typeof getInteractionStyles>,
  row: HTMLButtonElement
) => {
  await act(async () => {
    await focusWithKeyboard(row);
    await userEvent.hover(row);
    await waitForTransition();
  });

  expect(getInteractionStyles(row)).toEqual(expected);
};

const focusWithKeyboard = async (element: HTMLButtonElement) => {
  for (let index = 0; index < 12; index += 1) {
    if (document.activeElement === element) {
      return;
    }
    await userEvent.tab();
  }

  throw new Error(`Could not focus ${element.title} with the keyboard`);
};

const getInteractionStyles = (element: HTMLElement) => {
  const styles = getComputedStyle(element);

  return {
    backgroundColor: styles.backgroundColor,
    borderRadius: styles.borderRadius,
    outline: styles.outline,
    outlineOffset: styles.outlineOffset,
    transitionDuration: styles.transitionDuration,
    transitionProperty: styles.transitionProperty,
    transitionTimingFunction: styles.transitionTimingFunction,
  };
};

const waitForTransition = () =>
  new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, 200);
  });
