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
// Provide the Generation UI port for wildcard error reporting.
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() } }),
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
const pageErrors: string[] = [];
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const BASE_CONFIG: DynamicPromptsConfig = {
  combinatorial: true,
  maxPrompts: 100,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

/** Use stateful controlled values so toggles can round-trip. */
const render = async (onChange: (patch: Partial<DynamicPromptsConfig>) => void = vi.fn(), isSeedFixed = false) => {
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
              isSeedFixed,
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
  // Wait for scale animation before measuring geometry.
  await act(async () => {
    await new Promise((resolve) => {
      window.setTimeout(resolve, 300);
    });
  });
};

/** i18n is not bootstrapped in browser tests, so tabs are addressed by their stable id. */
const tabButton = (id: string) => document.querySelector<HTMLElement>(`[role="tab"][id$="-tab-${id}"]`);

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
  it('retires the seed sharing switch while the seed is held fixed', async () => {
    const onChange = vi.fn();

    await render(onChange, true);
    await openPopover();

    const seedSwitch = document.querySelector<HTMLElement>('[data-scope="switch"][data-part="root"]')!;
    const hiddenInput = seedSwitch.querySelector<HTMLInputElement>('input')!;
    const explanation = document.getElementById(hiddenInput.getAttribute('aria-describedby') ?? '');

    expect(hiddenInput.disabled).toBe(true);
    // No i18n instance here, so the key itself is the rendered text.
    expect(explanation?.textContent).toBe('widgets.generate.dynamicPrompts.seedHeldForEveryImage');

    // A retired switch ignores the label click that normally toggles it.
    await act(() => seedSwitch.querySelector<HTMLElement>('[data-part="label"]')?.click());

    expect(onChange).not.toHaveBeenCalled();
    expect(pageErrors).toEqual([]);
  });

  it('switches seed behaviour without tearing down the widget', async () => {
    // Cover hidden native-select synchronization failures.
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
    expect(tabButton('preview'), 'popover should still be mounted').toBeTruthy();
  });

  it('sends a click on the seed label to the switch, not the number input', async () => {
    // Clicking the label must toggle its own switch, not a sibling number input.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const label = document.querySelector<HTMLElement>('[data-scope="switch"][data-part="label"]')!;
    const switchRoot = document.querySelector('[data-scope="switch"][data-part="root"]')!;
    const hiddenInput = switchRoot.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    const numberInput = document.querySelector<HTMLInputElement>('[data-scope="number-input"][data-part="input"]')!;

    // Assert ID uniqueness directly, independently of prior render order.
    expect(hiddenInput.id).toBeTruthy();
    expect(hiddenInput.id).not.toBe(numberInput.id);

    await act(async () => {
      await userEvent.click(label);
    });

    expect(onChange).toHaveBeenCalledWith({ seedBehaviour: 'per-image' });
    expect(document.activeElement).not.toBe(numberInput);
  });

  it('keeps the settings row the same height in either mode', async () => {
    // Hidden shuffle still reserves space across modes.
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const row = () =>
      document.querySelector('[data-scope="menu"][data-part="trigger"]')!.closest('div')!.parentElement!;
    const combinatorialHeight = row().getBoundingClientRect().height;

    await act(async () => {
      await userEvent.click(document.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!);
    });
    // Dispatch clicks to test wiring without animation-stability waits; this does not test real pointer
    // actionability.
    await act(async () => {
      ([...document.querySelectorAll('[data-scope="menu"][data-part="item"]')].at(-1) as HTMLElement).click();
      await Promise.resolve();
    });

    expect(onChange).toHaveBeenCalledWith({ combinatorial: false });
    expect(row().getBoundingClientRect().height).toBe(combinatorialHeight);
  });

  it('labels both settings controls with a real, associated label', async () => {
    await render();
    await openPopover();

    const popover = document.querySelector('[data-scope="popover"][data-part="content"]')!;
    const numberInput = popover.querySelector<HTMLInputElement>('[data-scope="number-input"][data-part="input"]')!;
    const numberLabel = popover.querySelector<HTMLLabelElement>(`label[for="${numberInput.id}"]`);

    expect(numberLabel, 'number input should have an associated label').toBeTruthy();
    expect(numberInput.getAttribute('aria-label')).toBeNull();

    // The accessible name combines the field label and selected value.
    const trigger = popover.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!;
    const namedBy = (trigger.getAttribute('aria-labelledby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '');

    expect(namedBy.filter(Boolean)).toHaveLength(2);
  });

  it('anchors the mode menu to its trigger', async () => {
    // Supply IDs through Zag so popup anchoring resolves the trigger.
    await render();
    await openPopover();

    const trigger = document.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!;

    await act(async () => {
      trigger.click();
      await Promise.resolve();
    });
    await act(async () => {
      await new Promise((resolve) => {
        window.setTimeout(resolve, 300);
      });
    });

    const content = document.querySelector<HTMLElement>('[data-scope="menu"][data-part="content"]')!;
    const triggerBox = trigger.getBoundingClientRect();
    const contentBox = content.getBoundingClientRect();

    expect(contentBox.width).toBeGreaterThan(0);
    expect(Math.abs(contentBox.left - triggerBox.left)).toBeLessThan(40);
    expect(contentBox.top).toBeGreaterThan(triggerBox.top - 40);
  });

  it('gives both tabs a header control of the same height', async () => {
    await render();
    await openPopover();

    const headerControl = () =>
      document
        .querySelector('[data-scope="popover"][data-part="content"] p')!
        .parentElement!.lastElementChild!.getBoundingClientRect().height;

    const summaryHeight = headerControl();

    await act(async () => {
      await userEvent.click(tabButton('wildcards')!);
    });

    // Header geometry must remain equal across tabs.
    expect(headerControl()).toBe(summaryHeight);
  });
});
