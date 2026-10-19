/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { SeedMode } from '@platform/core/seed';

import { ChakraProvider } from '@chakra-ui/react';
import { planSeedSubmission } from '@platform/core/seed';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { Field } from './Field';
import { SeedInput, type SeedInputPatch } from './SeedInput';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

/** A host that owns the seed the way every widget does: the control only proposes patches. */
const Host = ({
  id,
  initialSeed,
  onCommit,
  seedMode,
}: {
  id?: string;
  initialSeed: number | undefined;
  onCommit: (patch: SeedInputPatch) => void;
  seedMode: SeedMode;
}) => {
  const [seed, setSeed] = useState(initialSeed);
  const plan =
    seedMode === 'increment' || seedMode === 'decrement'
      ? planSeedSubmission({
          batchCount: 3,
          promptCount: 1,
          seedBehaviour: 'per-iteration',
          seedMode,
          startSeed: seed ?? 0,
        })
      : null;

  return (
    <SeedInput
      ariaLabel="Seed"
      id={id}
      plan={plan}
      seed={seed}
      seedMode={seedMode}
      onCommit={(patch) => {
        if (patch.seed !== undefined) {
          setSeed(patch.seed);
        }

        onCommit(patch);
      }}
    />
  );
};

const renderHost = async (seedMode: SeedMode, initialSeed: number | undefined, id?: string) => {
  const onCommit = vi.fn();

  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <Host id={id} initialSeed={initialSeed} seedMode={seedMode} onCommit={onCommit} />
      </ChakraProvider>
    );
  });

  return onCommit;
};

/** Zag writes an external value into the input on an animation frame; wait two out so the DOM is settled. */
const settle = async (action: () => void) => {
  await act(async () => {
    action();
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
  });
};

const seedInput = () => host.querySelector<HTMLInputElement>('input[aria-label="Seed"]');
const dice = () => host.querySelector<HTMLButtonElement>('button[aria-label="common.newSeed"]');

describe('SeedInput', () => {
  it('shows the seed the dice rolled, not the one it replaced', async () => {
    const onCommit = await renderHost('fixed', 42);

    expect(seedInput()?.value).toBe('42');

    await settle(() => dice()?.click());

    const rolled = (onCommit.mock.calls[0]?.[0] as SeedInputPatch | undefined)?.seed as number;

    expect(Number.isInteger(rolled)).toBe(true);
    expect(rolled).not.toBe(42);
    // The host adopted the roll; the input has to show it, the same as any other external seed change.
    expect(seedInput()?.value).toBe(String(rolled));
  });

  it('shows the rolled seed under a host-supplied id too', async () => {
    // Host IDs must use Zag's ID map or external changes leave the DOM value stale.
    const onCommit = await renderHost('fixed', 42, 'node-1-seed');

    expect(seedInput()?.id).toBe('node-1-seed');

    await settle(() => dice()?.click());

    const rolled = (onCommit.mock.calls[0]?.[0] as SeedInputPatch | undefined)?.seed as number;

    expect(seedInput()?.value).toBe(String(rolled));
  });

  it('steps the seed with the stepper controls', async () => {
    const onCommit = await renderHost('fixed', 42);

    // Zag steps on press, so a real pointer sequence rather than a synthetic click.
    await act(async () => {
      await userEvent.click(host.querySelector<HTMLButtonElement>('[data-part="increment-trigger"]') as HTMLElement);
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    });

    expect(onCommit).toHaveBeenCalledWith({ seed: 43 });
    expect(seedInput()?.value).toBe('43');
  });

  it('commits only integers, rounding a typed fraction', async () => {
    const onCommit = await renderHost('fixed', 42);
    const input = seedInput() as HTMLInputElement;

    // Paste fractions as a whole; per-keystroke rounding removes a typed decimal point.
    await act(async () => {
      await userEvent.fill(input, '1.5');
      await new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      });
    });

    const committed = onCommit.mock.calls.map((call) => (call[0] as SeedInputPatch).seed);

    expect(committed.length).toBeGreaterThan(0);
    expect(committed.every((seed) => Number.isInteger(seed))).toBe(true);
    expect(committed.at(-1)).toBe(2);
    expect(input.value).toBe('2');
  });

  it('labels the stepper controls', async () => {
    await renderHost('fixed', 42);

    expect(host.querySelector('[data-part="increment-trigger"]')?.getAttribute('aria-label')).toBe(
      'common.increaseValue'
    );
    expect(host.querySelector('[data-part="decrement-trigger"]')?.getAttribute('aria-label')).toBe(
      'common.decreaseValue'
    );
  });

  it('keeps the entered seed on show in random mode while quieting the input and the dice', async () => {
    await renderHost('random', 42);

    expect(seedInput()?.value).toBe('42');
    expect(seedInput()?.disabled).toBe(true);
    expect(dice()?.disabled).toBe(true);
  });

  it('previews the batch and describes the input with it only in a stepping mode', async () => {
    await renderHost('increment', 42);

    const preview = host.querySelector<HTMLElement>('[data-testid="seed-sequence-preview"]');

    expect(preview?.textContent).toBe('common.seedNextBatchRange');
    expect(seedInput()?.getAttribute('aria-describedby')).toBe(preview?.id);

    await renderHost('fixed', 42);

    expect(host.querySelector('[data-testid="seed-sequence-preview"]')).toBeNull();
    expect(seedInput()?.getAttribute('aria-describedby')).toBeNull();
  });

  it("keeps the field's error in the input's description alongside the preview", async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field error="Out of range" label="Seed">
            <Host initialSeed={42} seedMode="increment" onCommit={vi.fn()} />
          </Field>
        </ChakraProvider>
      );
    });

    const errorId = host.querySelector('[data-part="error-text"]')?.id;
    const previewId = host.querySelector('[data-testid="seed-sequence-preview"]')?.id;
    const describedBy = seedInput()?.getAttribute('aria-describedby')?.split(' ') ?? [];

    expect(errorId).toBeTruthy();
    expect(describedBy).toContain(errorId);
    expect(describedBy).toContain(previewId);
  });

  it("keeps the field's error as the whole description while the seed holds", async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field error="Out of range" label="Seed">
            <Host initialSeed={42} seedMode="fixed" onCommit={vi.fn()} />
          </Field>
        </ChakraProvider>
      );
    });

    const errorId = host.querySelector('[data-part="error-text"]')?.id;

    expect(errorId).toBeTruthy();
    expect(host.querySelector('[data-testid="seed-sequence-preview"]')).toBeNull();
    expect(seedInput()?.getAttribute('aria-describedby')).toBe(errorId);
  });

  it("describes the input with the field's help text ahead of the preview", async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field helpText="Any whole number" label="Seed">
            <Host initialSeed={42} seedMode="increment" onCommit={vi.fn()} />
          </Field>
        </ChakraProvider>
      );
    });

    const helperId = host.querySelector('[data-part="helper-text"]')?.id;
    const previewId = host.querySelector('[data-testid="seed-sequence-preview"]')?.id;

    expect(helperId).toBeTruthy();
    expect(seedInput()?.getAttribute('aria-describedby')).toBe(`${helperId} ${previewId}`);
  });

  it('never names an error text the field does not render', async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field invalid label="Seed">
            <Host initialSeed={42} seedMode="fixed" onCommit={vi.fn()} />
          </Field>
        </ChakraProvider>
      );
    });

    expect(seedInput()?.getAttribute('aria-invalid')).toBe('true');
    expect(seedInput()?.getAttribute('aria-describedby')).toBeNull();
  });

  it('renders an empty field for an absent seed without committing anything', async () => {
    const onCommit = await renderHost('fixed', undefined);

    expect(seedInput()?.value).toBe('');
    expect(onCommit).not.toHaveBeenCalled();
  });
});
