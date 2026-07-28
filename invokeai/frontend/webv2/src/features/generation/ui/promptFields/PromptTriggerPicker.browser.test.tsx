/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

// The actions row and the picker's own contents need Generation's UI port; neither
// is what these tests are about. The stub records the two ways out of the picker so
// they can be invoked directly — a real click would have to compete with the
// textarea for the same coordinates.
const picker = vi.hoisted(() => ({ current: null as { onClose: () => void; onSelect: (t: string) => void } | null }));

vi.mock('@features/generation/ui/promptFields/PositivePromptActions', () => ({
  PositivePromptActions: () => null,
  PromptTriggerPopover: (props: { onClose: () => void; onSelect: (trigger: string) => void }) => {
    picker.current = props;
    return null;
  },
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

const render = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <PositivePromptField
            heightPx={120}
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
};

const textarea = () => host!.querySelector('textarea')!;
const isPickerOpen = () => picker.current !== null;

/** Runs one of the picker's exits, then lets insertPromptText's caret frame settle. */
const leavePicker = async (exit: (handlers: NonNullable<typeof picker.current>) => void) => {
  const handlers = picker.current!;

  await act(() => exit(handlers));
  await act(async () => {
    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });
};

const dismissPicker = () => leavePicker((handlers) => handlers.onClose());
const pickColors = () => leavePicker((handlers) => handlers.onSelect('__colors__'));

const type = async (text: string) => {
  await act(async () => {
    await userEvent.click(textarea());
  });
  await act(async () => {
    await userEvent.type(textarea(), text);
  });
};

beforeEach(async () => {
  picker.current = null;
  await render();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the prompt trigger picker and the keystroke that opens it', () => {
  // Regression: the picker opened on any second underscore and swallowed the
  // keystroke, so an ordinary word containing `__` could not be typed at all.
  it('leaves an underscore inside a word alone', async () => {
    await type('a close_up shot');

    expect(textarea().value).toBe('a close_up shot');
    expect(isPickerOpen()).toBe(false);
  });

  it('types a double underscore mid-word without opening the picker', async () => {
    await type('snake__case');

    expect(textarea().value).toBe('snake__case');
    expect(isPickerOpen()).toBe(false);
  });

  it('opens on `__` where a wildcard reference could begin', async () => {
    await type('a photo of __');

    expect(isPickerOpen()).toBe(true);
  });

  // Regression: dismissing left the prompt one underscore short of what was typed.
  it('gives the underscores back when the picker is dismissed', async () => {
    await type('a photo of __');
    await dismissPicker();

    expect(textarea().value).toBe('a photo of __');
  });

  it('gives `<` back when the picker is dismissed', async () => {
    await type('a photo of <');
    expect(isPickerOpen()).toBe(true);

    await dismissPicker();

    expect(textarea().value).toBe('a photo of <');
  });

  it('replaces the typed underscores with the picked reference', async () => {
    await type('a photo of __');
    await pickColors();

    expect(textarea().value).toBe('a photo of __colors__');
  });
});
