/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('@features/generation/ui/promptFields/PositivePromptActions', () => ({
  PositivePromptActions: () => null,
  PromptTriggerPopover: () => null,
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ models: { catalog: [], ensureLoaded: vi.fn() } }),
}));

/** Enough wildcards that the suggestion list overflows its max height and grows a scrollbar. */
const WILDCARDS = Array.from({ length: 30 }, (_, index) => ({
  id: `w${index + 1}`,
  name: `wild${String(index + 1).padStart(2, '0')}`,
  values: ['x'],
}));

vi.mock('@features/generation/data/wildcards', async (importOriginal) => ({
  // Partial: the real module's other exports (constants, mappers) stay available to what imports them.
  ...(await importOriginal<Record<string, unknown>>()),
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve(WILDCARDS), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          {/* Match the prompt box's production drop-target environment with DndContext. */}
          <DndContext>
            <PositivePromptField
              heightPx={320}
              loras={[]}
              projectId="project-1"
              selectedModel={undefined}
              showSyntaxHighlighting={false}
              value=""
              onChange={vi.fn()}
              onResizeEnd={vi.fn()}
              onUsePrompt={vi.fn()}
            />
          </DndContext>
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });
};

const textarea = () => host!.querySelector('textarea')!;
const listbox = () => document.querySelector<HTMLElement>('[role="listbox"]');

const type = async (text: string) => {
  await act(async () => {
    await userEvent.click(textarea());
  });
  await act(async () => {
    await userEvent.type(textarea(), text);
  });
};

const press = async (key: string) => {
  await act(async () => {
    await userEvent.keyboard(key);
  });
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
};

beforeEach(async () => {
  await render();
  await type('a photo of __');
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the caret autocomplete list', () => {
  it('renders a scrollable list of every match', () => {
    expect(listbox()).not.toBeNull();
    expect(listbox()!.querySelectorAll('[role="option"]')).toHaveLength(30);
    expect(listbox()!.scrollHeight).toBeGreaterThan(listbox()!.clientHeight);
  });

  it('stays open when the list itself scrolls, as with the wheel or scrollbar', async () => {
    const list = listbox()!;

    await act(() => {
      list.scrollTop = 60;
      list.dispatchEvent(new Event('scroll'));
    });

    expect(listbox()).toBe(list);
    expect(textarea().getAttribute('aria-expanded')).toBe('true');
  });

  it('keeps its place when arrowing past the visible end of the list', async () => {
    for (let index = 0; index < 10; index++) {
      await press('{ArrowDown}');
    }

    const list = listbox();

    expect(list).not.toBeNull();
    expect(list!.scrollTop).toBeGreaterThan(0);

    const activeId = textarea().getAttribute('aria-activedescendant');
    const options = [...list!.querySelectorAll('[role="option"]')];

    expect(activeId).toBe(options[10]!.id);
    expect(options[10]!.getAttribute('aria-selected')).toBe('true');
  });
});
