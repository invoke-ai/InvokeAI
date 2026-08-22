import { Box, ChakraProvider, Stack, Tabs } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

const variants = ['line', 'subtle', 'enclosed', 'outline', 'plain'] as const;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/**
 * This test drives real pointer input across five variants, and each move costs
 * whatever the machine can spare. Measured on a loaded 16-core box, the moves
 * alone took 7–13.5s against the default 15s, and every failure was a timeout
 * sitting at 12.5–13.6s — never an assertion. The settle waits were flat at
 * ~1.1s throughout, so synchronization was never the problem; the budget was.
 * Dropping the redundant unhovers below removes a third of the moves, and this
 * ceiling covers what is left.
 */
const HOVER_SWEEP_TIMEOUT_MS = 60_000;

describe('tab hover styles', () => {
  it(
    'gives every variant restrained hover feedback without changing selected or disabled tabs',
    async () => {
      host = document.createElement('div');
      document.body.append(host);
      root = createRoot(host);

      await act(async () => {
        root?.render(
          <ChakraProvider value={system}>
            <Stack>
              <Box aria-label="muted background probe" bg="bg.muted" />
              <Box aria-label="faint muted background probe" bg="bg.muted/60" />
              <Box aria-label="light muted background probe" bg="bg.muted/40" />
              <Box aria-label="emphasized background probe" bg="bg.emphasized" />
              <Box aria-label="emphasized border probe" borderColor="border.emphasized" borderWidth="1px" />
              {variants.map((variant) => (
                <Tabs.Root key={variant} value="selected" variant={variant}>
                  <Tabs.List>
                    <Tabs.Trigger aria-label={`${variant} idle`} value="idle">
                      Idle
                    </Tabs.Trigger>
                    <Tabs.Trigger aria-label={`${variant} selected`} value="selected">
                      Selected
                    </Tabs.Trigger>
                    <Tabs.Trigger aria-label={`${variant} disabled`} value="disabled" disabled>
                      Disabled
                    </Tabs.Trigger>
                  </Tabs.List>
                </Tabs.Root>
              ))}
            </Stack>
          </ChakraProvider>
        );
        await new Promise<void>((resolve) => {
          globalThis.setTimeout(resolve, 0);
        });
      });

      const mutedBackground = getProbeStyle(host, 'muted background probe').backgroundColor;
      const faintMutedBackground = getProbeStyle(host, 'faint muted background probe').backgroundColor;
      const lightMutedBackground = getProbeStyle(host, 'light muted background probe').backgroundColor;
      const emphasizedBackground = getProbeStyle(host, 'emphasized background probe').backgroundColor;
      const emphasizedBorder = getProbeStyle(host, 'emphasized border probe').borderColor;
      const expectedHoverBackgrounds = {
        enclosed: emphasizedBackground,
        line: faintMutedBackground,
        outline: mutedBackground,
        plain: lightMutedBackground,
        subtle: mutedBackground,
      };

      for (const variant of variants) {
        const idle = host.querySelector<HTMLButtonElement>(`[aria-label="${variant} idle"]`)!;
        const selected = host.querySelector<HTMLButtonElement>(`[aria-label="${variant} selected"]`)!;
        const disabled = host.querySelector<HTMLButtonElement>(`[aria-label="${variant} disabled"]`)!;
        const idleBefore = getInteractionStyles(idle);
        const selectedBefore = getInteractionStyles(selected);
        const disabledBefore = getInteractionStyles(disabled);

        await act(async () => {
          await userEvent.hover(idle);
          await waitForSettledStyles(idle);
        });
        const idleHovered = getInteractionStyles(idle);
        expect(idleHovered.transitionDuration).toBe('0.1s');
        expect(idleHovered.transitionProperty).toBe('background, border-color, color');
        expect(idleHovered.backgroundColor).toBe(expectedHoverBackgrounds[variant]);
        if (variant === 'line' || variant === 'plain') {
          expect(idleHovered.color).not.toBe(idleBefore.color);
        } else {
          expect(idleHovered.color).toBe(idleBefore.color);
        }
        if (variant === 'outline') {
          expect(idleHovered.borderColor).toBe(emphasizedBorder);
        }

        await act(async () => {
          await userEvent.hover(selected);
          await waitForSettledStyles(selected);
        });
        expect(getInteractionStyles(selected)).toEqual(selectedBefore);

        await act(async () => {
          await userEvent.hover(disabled);
          await waitForSettledStyles(disabled);
        });
        expect(getInteractionStyles(disabled)).toEqual(disabledBefore);
      }

      const lineIdle = host.querySelector<HTMLButtonElement>('[aria-label="line idle"]')!;
      const lineSelected = host.querySelector<HTMLButtonElement>('[aria-label="line selected"]')!;
      await act(async () => {
        await userEvent.tab();
        await userEvent.keyboard('{ArrowLeft}');
      });
      expect(document.activeElement).toBe(lineIdle);
      const focusOutline = getComputedStyle(lineIdle).outline;
      expect(focusOutline).not.toBe('none');

      await act(async () => {
        await userEvent.hover(lineIdle);
        await waitForSettledStyles(lineIdle);
      });
      expect(getComputedStyle(lineIdle).outline).toBe(focusOutline);
      expect(lineSelected.dataset.selected).toBe('');
    },
    HOVER_SWEEP_TIMEOUT_MS
  );
});

const getInteractionStyles = (element: HTMLElement) => {
  const styles = getComputedStyle(element);

  return {
    backgroundColor: styles.backgroundColor,
    borderColor: styles.borderColor,
    color: styles.color,
    transitionDuration: styles.transitionDuration,
    transitionProperty: styles.transitionProperty,
  };
};

/** Consecutive unchanged samples that count as "the transition has finished". */
const STABLE_SAMPLES = 3;
const SAMPLE_INTERVAL_MS = 16;

/**
 * Waits until an element's interaction styles stop changing, rather than
 * sleeping a fixed interval after each hover.
 *
 * These transitions run for 0.1s, but the test hovers fifteen triggers and
 * used to wait 200ms every time — three seconds of sleeping against a 15s
 * test timeout, which is what made this fail on a loaded CI runner. Sampling
 * until the values hold still takes as long as the machine actually needs,
 * and usually far less.
 */
const waitForSettledStyles = (element: HTMLElement, timeoutMs = 5000): Promise<void> =>
  new Promise<void>((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    let previous = JSON.stringify(getInteractionStyles(element));
    let stableSamples = 0;

    const sample = () => {
      const current = JSON.stringify(getInteractionStyles(element));

      stableSamples = current === previous ? stableSamples + 1 : 0;
      previous = current;

      if (stableSamples >= STABLE_SAMPLES) {
        resolve();

        return;
      }

      if (Date.now() > deadline) {
        reject(new Error(`Timed out after ${timeoutMs}ms waiting for styles to settle; last value ${current}`));

        return;
      }

      globalThis.setTimeout(sample, SAMPLE_INTERVAL_MS);
    };

    globalThis.setTimeout(sample, SAMPLE_INTERVAL_MS);
  });

const getProbeStyle = (container: HTMLElement, label: string) =>
  getComputedStyle(container.querySelector<HTMLElement>(`[aria-label="${label}"]`)!);
