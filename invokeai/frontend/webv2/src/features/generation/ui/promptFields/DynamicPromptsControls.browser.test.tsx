/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { DynamicPromptsConfig } from '@features/generation/core/dynamicPrompts';
import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { ChakraProvider } from '@chakra-ui/react';
import { DynamicPromptsButton } from '@features/generation/ui/promptFields/DynamicPromptsButton';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/data/promptUtilities', () => ({ parseDynamicPrompts }));
vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({ notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() } }),
}));
vi.mock('@features/generation/data/wildcards', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
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
  enabled: true,
  maxPrompts: 100,
  resample: true,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

const PROMPT = 'a {red|green} cat';

const NO_PROMPT_ELEMENT = { element: () => null };

const ExpandingButton = ({ config }: { config: DynamicPromptsFieldConfig }) => (
  <DynamicPromptsButton
    batchCount={2}
    config={config}
    expansion={useDynamicPrompts(PROMPT, config)}
    highlightExpandedPrompts
    promptEditor={NO_PROMPT_ELEMENT}
    showSyntaxHighlighting
    onInsertText={vi.fn()}
    onUsePrompt={vi.fn()}
  />
);

const switchRoot = (label: string) =>
  [...document.querySelectorAll<HTMLElement>('[data-scope="switch"][data-part="root"]')].find(
    (root) => root.querySelector('[data-part="label"]')?.textContent === label
  )!;
const SEED_SWITCH = 'widgets.generate.dynamicPrompts.newSeedPerImage';

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
          <ExpandingButton
            config={{
              ...config,
              isSeedFixed,
              onChange: (patch) => {
                onChange(patch);
                config = { ...config, ...patch };
                void act(() => draw());
              },
            }}
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
  it('turns dynamic prompts off, leaving only the switch and no expansion', async () => {
    const onChange = vi.fn();

    await render(onChange);
    await vi.waitFor(() => expect(parseDynamicPrompts).toHaveBeenCalledOnce());
    await openPopover();

    await act(async () => {
      await userEvent.click(switchRoot('widgets.generate.dynamicPrompts.enabled'));
    });

    expect(onChange).toHaveBeenCalledWith({ enabled: false });
    expect(switchRoot(SEED_SWITCH), 'the expansion settings go with it').toBeUndefined();
    expect(document.body.textContent).toContain('widgets.generate.dynamicPrompts.disabledHelp');
    expect(document.body.textContent).not.toContain('a green cat');
    expect([...host!.querySelectorAll('button')].at(-1)!.textContent).toContain('widgets.generate.dynamicPrompts.off');
    expect(parseDynamicPrompts).toHaveBeenCalledOnce();
  });

  it('retires the seed sharing switch while the seed is held fixed', async () => {
    const onChange = vi.fn();

    await render(onChange, true);
    await openPopover();

    const seedSwitch = switchRoot(SEED_SWITCH);
    const hiddenInput = seedSwitch.querySelector<HTMLInputElement>('input')!;
    const explanation = document.getElementById(hiddenInput.getAttribute('aria-describedby') ?? '');

    expect(hiddenInput.disabled).toBe(true);
    expect(explanation?.textContent).toBe('widgets.generate.dynamicPrompts.seedHeldForEveryImage');

    await act(() => seedSwitch.querySelector<HTMLElement>('[data-part="label"]')?.click());

    expect(onChange).not.toHaveBeenCalled();
    expect(pageErrors).toEqual([]);
  });

  it('switches seed behaviour without tearing down the widget', async () => {
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const seedSwitch = switchRoot(SEED_SWITCH);

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
    const onChange = vi.fn();

    await render(onChange);
    await openPopover();

    const seedSwitch = switchRoot(SEED_SWITCH);
    const label = seedSwitch.querySelector<HTMLElement>('[data-part="label"]')!;
    const hiddenInput = seedSwitch.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    const numberInput = document.querySelector<HTMLInputElement>('[data-scope="number-input"][data-part="input"]')!;

    expect(hiddenInput.id).toBeTruthy();
    expect(hiddenInput.id).not.toBe(numberInput.id);

    await act(async () => {
      await userEvent.click(label);
    });

    expect(onChange).toHaveBeenCalledWith({ seedBehaviour: 'per-image' });
    expect(document.activeElement).not.toBe(numberInput);
  });

  it('offers resampling only for random samples', async () => {
    const onChange = vi.fn();
    const resampleLabel = () =>
      [...document.querySelectorAll<HTMLElement>('[data-scope="switch"][data-part="label"]')].find(
        (label) => label.textContent === 'widgets.generate.dynamicPrompts.newSampleEveryInvoke'
      );

    await render(onChange);
    await openPopover();

    expect(resampleLabel(), 'all combinations have nothing to resample').toBeUndefined();

    await act(async () => {
      await userEvent.click(document.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!);
    });
    await act(async () => {
      ([...document.querySelectorAll('[data-scope="menu"][data-part="item"]')].at(-1) as HTMLElement).click();
      await Promise.resolve();
    });

    expect(onChange).toHaveBeenCalledWith({ combinatorial: false });
    expect(resampleLabel(), 'random sample offers resampling').toBeTruthy();

    await act(async () => {
      await userEvent.click(resampleLabel()!);
    });

    expect(onChange).toHaveBeenCalledWith({ resample: false });
    expect(pageErrors).toEqual([]);
  });

  it('labels both settings controls with a real, associated label', async () => {
    await render();
    await openPopover();

    const popover = document.querySelector('[data-scope="popover"][data-part="content"]')!;
    const numberInput = popover.querySelector<HTMLInputElement>('[data-scope="number-input"][data-part="input"]')!;
    const numberLabel = popover.querySelector<HTMLLabelElement>(`label[for="${numberInput.id}"]`);

    expect(numberLabel, 'number input should have an associated label').toBeTruthy();
    expect(numberInput.getAttribute('aria-label')).toBeNull();

    const trigger = popover.querySelector<HTMLElement>('[data-scope="menu"][data-part="trigger"]')!;
    const namedBy = (trigger.getAttribute('aria-labelledby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '');

    expect(namedBy.filter(Boolean)).toHaveLength(2);
  });

  it('anchors the mode menu to its trigger', async () => {
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
});
