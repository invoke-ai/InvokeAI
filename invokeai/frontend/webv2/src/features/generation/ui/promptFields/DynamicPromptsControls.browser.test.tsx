/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';

import { ChakraProvider } from '@chakra-ui/react';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
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
const pageErrors: string[] = [];
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const BASE_CONFIG: DynamicPromptsConfig = {
  combinatorial: true,
  maxPrompts: 100,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

/**
 * Renders with real state, so a control that reflects its own value (a segmented
 * control does) can actually be toggled back and forth.
 */
const render = async (onChange: (patch: Partial<DynamicPromptsConfig>) => void = vi.fn()) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const client = new QueryClient();
  let config = BASE_CONFIG;

  const draw = () =>
    root?.render(
      <QueryClientProvider client={client}>
        <ChakraProvider value={system}>
          <DynamicPromptsButton
            batchCount={2}
            config={{
              ...config,
              onChange: (patch) => {
                onChange(patch);
                config = { ...config, ...patch };
                void act(() => draw());
              },
            }}
            positivePrompt="a {red|green} cat"
            showSyntaxHighlighting
            onInsertText={vi.fn()}
            onUsePrompt={vi.fn()}
          />
        </ChakraProvider>
      </QueryClientProvider>
    );

  await act(() => draw());
};

const openPopover = async () => {
  await act(async () => {
    await userEvent.click([...host!.querySelectorAll('button')].at(-1)!);
  });
  // The popover opens with a scale transform, and getBoundingClientRect includes
  // transforms — measuring before it settles reports every box ~5% short.
  await act(async () => {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 300);
    });
  });
};

/**
 * Segment labels intercept pointer events, so the visible text is the click
 * target. i18n is not bootstrapped in browser tests, so labels render as keys —
 * segments are addressed by their stable value instead.
 */
const segmentLabel = (value: string) =>
  document.querySelector(`[data-scope="segment-group"][data-part="item-text"][id$=":radio:label:${value}"]`);

const captureError = (event: ErrorEvent) => pageErrors.push(event.message);

beforeEach(() => {
  pageErrors.length = 0;
  window.addEventListener('error', captureError);
  parseDynamicPrompts.mockReset();
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['a red cat', 'a green cat'] });
});

afterEach(async () => {
  window.removeEventListener('error', captureError);
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('dynamic prompts popover controls', () => {
  it('switches seed behaviour without tearing down the widget', async () => {
    // Regression: a Chakra Select here threw `r.options is not iterable` from
    // syncSelectElement as the popover opened, taking the Generate widget down
    // with it. The seed choice is a switch now, which has no hidden native
    // select to sync.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const seedSwitch = document.querySelector<HTMLElement>('[data-scope="switch"][data-part="root"]')!;

    expect(seedSwitch, 'seed switch should be rendered').toBeTruthy();

    for (const expected of ['per-image', 'per-iteration'] as const) {
      await act(async () => {
        await userEvent.click(seedSwitch);
      });
      expect(onChange).toHaveBeenCalledWith({ seedBehaviour: expected });
    }

    expect(pageErrors).toEqual([]);
    expect(segmentLabel('preview'), 'popover should still be mounted').toBeTruthy();
  });

  it('sends a click on the seed label to the switch, not the number input', async () => {
    // Regression: Chakra derives the switch's label `for` from its own id
    // counter, which collided with the number input beside it, so clicking the
    // label focused Max prompts instead of toggling the switch.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const label = document.querySelector<HTMLElement>('[data-scope="switch"][data-part="label"]')!;
    const switchRoot = document.querySelector('[data-scope="switch"][data-part="root"]')!;
    const hiddenInput = switchRoot.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    const numberInput = document.querySelector<HTMLInputElement>('[data-scope="number-input"][data-part="input"]')!;

    // Whether the generated ids actually collide depends on how many other
    // controls rendered first, so this asserts the condition that caused it —
    // two controls sharing an id — rather than relying on the ordering that
    // happened to trip it in the app.
    expect(hiddenInput.id).toBeTruthy();
    expect(hiddenInput.id).not.toBe(numberInput.id);

    await act(async () => {
      await userEvent.click(label);
    });

    expect(onChange).toHaveBeenCalledWith({ seedBehaviour: 'per-image' });
    expect(document.activeElement).not.toBe(numberInput);
  });

  it('keeps the settings row the same height in either mode', async () => {
    // The shuffle button only applies to a random sample, so it is always
    // rendered and merely hidden; switching modes must not reflow the row.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const row = () =>
      document.querySelector('[data-scope="menu"][data-part="trigger"]')!.closest('div')!.parentElement!;
    const combinatorialHeight = row().getBoundingClientRect().height;

    await act(async () => {
      await userEvent.click(document.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!);
    });
    await act(async () => {
      await userEvent.click(
        [...document.querySelectorAll('[data-scope="menu"][data-part="item"]')].at(-1) as HTMLElement
      );
    });

    expect(onChange).toHaveBeenCalledWith({ combinatorial: false });
    expect(row().getBoundingClientRect().height).toBe(combinatorialHeight);
  });

  it('keeps the tabs to their content width rather than stretching them', async () => {
    await render(vi.fn());
    await openPopover();

    const tabs = segmentLabel('preview')!.closest('[data-scope="segment-group"][data-part="root"]')!;

    // Asserting the computed alignment rather than a pixel width: browser tests
    // render i18n keys, so the labels are not their real size here.
    expect(getComputedStyle(tabs).alignSelf).toBe('start');
  });
});
