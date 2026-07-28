/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// The actions row needs Generation's UI port and is not what these tests are
// about; the caret autocomplete underneath it is entirely real.
vi.mock('@features/generation/ui/promptFields/PositivePromptActions', () => ({
  PositivePromptActions: () => null,
  PromptTriggerPopover: () => null,
}));

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ models: { catalog: [], ensureLoaded: vi.fn() } }),
}));

const WILDCARDS = [
  { id: 'w1', name: 'colors', values: ['red'] },
  { id: 'w2', name: 'moods', values: ['calm'] },
];

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve(WILDCARDS), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Tall on purpose: the old picker anchored to the whole box, which is the bug. */
const TEXTAREA_HEIGHT_PX = 320;

const render = async (isTemplateViewMode = false) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <PositivePromptField
            heightPx={TEXTAREA_HEIGHT_PX}
            isTemplateViewMode={isTemplateViewMode}
            loras={[]}
            projectId="project-1"
            selectedModel={undefined}
            showSyntaxHighlighting={false}
            value=""
            onChange={vi.fn()}
            onResizeEnd={vi.fn()}
            onUsePrompt={vi.fn()}
          />
        </ChakraProvider>
      </QueryClientProvider>
    );
  });

  // The option list is fed by the wildcard query, which resolves a tick later.
  await act(async () => {
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });
};

const textarea = () => host!.querySelector('textarea')!;
const listbox = () => document.querySelector('[role="listbox"]');
const optionLabels = () => [...(listbox()?.querySelectorAll('[role="option"]') ?? [])].map((o) => o.textContent);

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
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the caret autocomplete', () => {
  // Regression: the picker opened on any second underscore and swallowed the
  // keystroke, so an ordinary word containing `__` could not be typed at all.
  it('leaves an underscore inside a word alone', async () => {
    await type('a close_up shot');

    expect(textarea().value).toBe('a close_up shot');
    expect(listbox()).toBeNull();
  });

  it('types a double underscore mid-word without opening', async () => {
    await type('snake__case');

    expect(textarea().value).toBe('snake__case');
    expect(listbox()).toBeNull();
  });

  it('opens on `__` where a wildcard reference could begin', async () => {
    await type('a photo of __');

    expect(optionLabels()).toEqual(['colors', 'moods']);
  });

  // The whole point: nothing is swallowed, so there is nothing to give back.
  it('leaves the typed characters in the prompt while it is open', async () => {
    await type('a photo of __');

    expect(textarea().value).toBe('a photo of __');
  });

  it('narrows as the name is typed, where the user is already looking', async () => {
    await type('a photo of __mo');

    expect(optionLabels()).toEqual(['moods']);
  });

  it('closes when nothing matches, rather than showing an empty list', async () => {
    await type('a photo of __zzz');

    expect(listbox()).toBeNull();
    expect(textarea().value).toBe('a photo of __zzz');
  });

  it('opens beside the caret, not at the bottom of a tall box', async () => {
    await type('a photo of __');

    const box = textarea().getBoundingClientRect();
    const list = listbox()!.getBoundingClientRect();

    expect(box.height).toBeGreaterThan(200);
    // On the first line of the prompt, so the list belongs at the top of the box.
    expect(list.top - box.top).toBeLessThan(80);
  });

  // The complaint this whole thing exists to answer: the list used to open at
  // the bottom-left of the box no matter where you were typing.
  it('follows the caret down the box', async () => {
    await type('one\ntwo\nthree\nfour\nfive\nsix __');

    const box = textarea().getBoundingClientRect();
    const list = listbox()!.getBoundingClientRect();

    expect(list.top - box.top).toBeGreaterThan(60);
  });

  it('inserts the highlighted option on Enter', async () => {
    await type('a photo of __');
    await press('{Enter}');

    expect(textarea().value).toBe('a photo of __colors__');
    expect(listbox()).toBeNull();
  });

  it('moves the highlight with the arrow keys', async () => {
    await type('a photo of __');
    await press('{ArrowDown}');
    await press('{Enter}');

    expect(textarea().value).toBe('a photo of __moods__');
  });

  it('wraps the highlight around the ends of the list', async () => {
    await type('a photo of __');
    await press('{ArrowUp}');
    await press('{Enter}');

    expect(textarea().value).toBe('a photo of __moods__');
  });

  it('replaces what was typed, not just the trigger', async () => {
    await type('a photo of __mo');
    await press('{Enter}');

    expect(textarea().value).toBe('a photo of __moods__');
  });

  it('closes on Escape and leaves the prompt exactly as typed', async () => {
    await type('a photo of __');
    await press('{Escape}');

    expect(listbox()).toBeNull();
    expect(textarea().value).toBe('a photo of __');
  });

  it('tells assistive tech the textarea is a combobox over the list', async () => {
    await type('a photo of __');

    const activeId = textarea().getAttribute('aria-activedescendant');

    expect(textarea().getAttribute('role')).toBe('combobox');
    expect(textarea().getAttribute('aria-expanded')).toBe('true');
    expect(textarea().getAttribute('aria-controls')).toBe(listbox()!.id);
    expect(document.getElementById(activeId!)?.getAttribute('aria-selected')).toBe('true');
  });

  it('closes once the reference is finished', async () => {
    await type('a photo of __colors__');

    expect(listbox()).toBeNull();
  });

  // Regression: the template view-mode toggle can be on with no template
  // applied, which leaves the prompt fully editable. Gating on the toggle alone
  // killed the autocomplete outright while the user was still typing.
  it('still opens when view mode is on but no template is applied', async () => {
    await act(() => root?.unmount());
    host?.remove();
    await render(true);
    await type('a photo of __');

    expect(textarea().readOnly).toBe(false);
    expect(optionLabels()).toEqual(['colors', 'moods']);
  });
});
