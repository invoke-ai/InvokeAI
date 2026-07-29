/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
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

const MODEL_CATALOG = [{ base: 'sdxl', name: 'easynegative', type: 'embedding' }];
const SELECTED_MODEL = { base: 'sdxl', name: 'Juggernaut', trigger_phrases: ['jugg'] };

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ models: { catalog: MODEL_CATALOG, ensureLoaded: vi.fn() } }),
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
            selectedModel={SELECTED_MODEL as never}
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
  it('offers embeddings for `<` and wildcards for `__`, and phrases for neither', async () => {
    await type('a photo of <');

    expect(optionLabels()).toEqual(['easynegative']);

    await press('{Escape}');
    await type('__');

    expect(optionLabels()).toEqual(['colors', 'moods']);
  });

  it('narrows as the name is typed, where the user is already looking', async () => {
    await type('a photo of __mo');

    expect(optionLabels()).toEqual(['moods']);
  });

  it('opens beside the caret, not at the bottom of a tall box', async () => {
    await type('a photo of __');

    const box = textarea().getBoundingClientRect();
    const list = listbox()!.getBoundingClientRect();

    expect(box.height).toBeGreaterThan(200);
    expect(list.top - box.top).toBeLessThan(80);
  });

  it('moves the highlight with the arrow keys', async () => {
    await type('a photo of __');
    await press('{ArrowDown}');
    await press('{Enter}');

    expect(textarea().value).toBe('a photo of __moods__');
  });

  const mouseDownOnFirstOption = (button: number) => {
    const option = listbox()!.querySelector('[role="option"]')!;

    return act(() => {
      option.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button, cancelable: true }));
    });
  };

  it('keeps the rest of the prompt when the pointer selects', async () => {
    await type('a photo of __co');
    const activeElement = vi.spyOn(document, 'activeElement', 'get').mockReturnValue(document.body);
    await mouseDownOnFirstOption(0);
    activeElement.mockRestore();

    expect(textarea().value).toBe('a photo of __colors__');
  });

  describe('while an IME is composing', () => {
    const compose = (type: 'compositionend' | 'compositionstart') =>
      act(() => {
        textarea().dispatchEvent(new CompositionEvent(type, { bubbles: true, data: 'ねこ' }));
      });

    const pressWhileComposing = (key: string) =>
      act(() => {
        textarea().dispatchEvent(
          new KeyboardEvent('keydown', { bubbles: true, cancelable: true, isComposing: true, key })
        );
      });

    it('leaves navigation and commit keys to the IME, then reopens on composition end', async () => {
      await type('a photo of __');
      await compose('compositionstart');
      expect(listbox()).toBeNull();

      await pressWhileComposing('Enter');
      await pressWhileComposing('ArrowDown');
      expect(textarea().value).toBe('a photo of __');

      await compose('compositionend');
      expect(optionLabels()).toEqual(['colors', 'moods']);

      await press('{Enter}');
      expect(textarea().value).toBe('a photo of __colors__');
    });
  });

  it('closes when something scrolls under it', async () => {
    await type('a photo of __');

    expect(listbox()).not.toBeNull();

    await act(() => {
      textarea().dispatchEvent(new Event('scroll'));
    });

    expect(listbox()).toBeNull();
    expect(textarea().value).toBe('a photo of __');
  });

  it('closes when the window is resized', async () => {
    await type('a photo of __');

    await act(() => {
      window.dispatchEvent(new Event('resize'));
    });

    expect(listbox()).toBeNull();
  });

  it('tells assistive tech the textarea is a combobox over the list', async () => {
    await type('a photo of __');

    const activeId = textarea().getAttribute('aria-activedescendant');

    expect(textarea().getAttribute('role')).toBe('combobox');
    expect(textarea().getAttribute('aria-expanded')).toBe('true');
    expect(textarea().getAttribute('aria-controls')).toBe(listbox()!.id);
    expect(document.getElementById(activeId!)?.getAttribute('aria-selected')).toBe('true');
  });

  it('still opens when view mode is on but no template is applied', async () => {
    await act(() => root?.unmount());
    host?.remove();
    await render(true);
    await type('a photo of __');

    expect(textarea().readOnly).toBe(false);
    expect(optionLabels()).toEqual(['colors', 'moods']);
  });
});
