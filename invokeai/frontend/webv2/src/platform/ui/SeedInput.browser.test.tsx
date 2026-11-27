/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { SeedMode } from '@platform/core/seed';

import { ChakraProvider } from '@chakra-ui/react';
import { planSeedSubmission } from '@platform/core/seed';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

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

  it('previews the batch and describes the input with it only in a stepping mode', async () => {
    await renderHost('increment', 42);

    const preview = host.querySelector<HTMLElement>('[data-testid="seed-sequence-preview"]');

    expect(preview?.textContent).toBe('common.seedNextBatchRange');
    expect(seedInput()?.getAttribute('aria-describedby')).toBe(preview?.id);

    await renderHost('fixed', 42);

    expect(host.querySelector('[data-testid="seed-sequence-preview"]')).toBeNull();
    expect(seedInput()?.getAttribute('aria-describedby')).toBeNull();
  });
});
