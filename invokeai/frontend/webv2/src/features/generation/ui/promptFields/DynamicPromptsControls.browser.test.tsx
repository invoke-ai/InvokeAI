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
    // with it. A two-option choice is a segmented control now.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    for (const expected of ['per-image', 'per-iteration'] as const) {
      const target = segmentLabel(expected);

      expect(target, `${expected} should be rendered`).toBeTruthy();
      await act(async () => {
        await userEvent.click(target!);
      });
      expect(onChange).toHaveBeenCalledWith({ seedBehaviour: expected });
    }

    expect(pageErrors).toEqual([]);
    expect(segmentLabel('preview'), 'popover should still be mounted').toBeTruthy();
  });

  it('never lets the shuffle button drive the mode row height', async () => {
    // The button used to be ~8px taller than the segmented control beside it, so
    // switching to Random visibly nudged the row. It is now always rendered (just
    // hidden) and never taller than the control, so the row is governed by the
    // segmented control in both modes.
    await render();
    await openPopover();

    const segments = () => segmentLabel('all')!.closest('[data-scope="segment-group"][data-part="root"]')!;
    const row = () => segments().parentElement!;
    const heights = () => [row().getBoundingClientRect().height, segments().getBoundingClientRect().height];

    const [combinatorialRow, combinatorialSegments] = heights();

    expect(combinatorialRow).toBe(combinatorialSegments);

    await act(async () => {
      await userEvent.click(segmentLabel('random')!);
    });

    const [randomRow, randomSegments] = heights();

    expect(randomRow).toBe(randomSegments);
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
