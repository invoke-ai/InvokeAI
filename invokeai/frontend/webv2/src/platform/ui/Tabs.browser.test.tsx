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

describe('tab hover styles', () => {
  it('gives every variant restrained hover feedback without changing selected or disabled tabs', async () => {
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
        await waitForTransition();
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
        await userEvent.unhover(idle);
        await userEvent.hover(selected);
        await waitForTransition();
      });
      expect(getInteractionStyles(selected)).toEqual(selectedBefore);

      await act(async () => {
        await userEvent.unhover(selected);
        await userEvent.hover(disabled);
        await waitForTransition();
      });
      expect(getInteractionStyles(disabled)).toEqual(disabledBefore);
      await act(() => userEvent.unhover(disabled));
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
      await waitForTransition();
    });
    expect(getComputedStyle(lineIdle).outline).toBe(focusOutline);
    expect(lineSelected.dataset.selected).toBe('');
  });
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

const waitForTransition = () =>
  new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, 200);
  });

const getProbeStyle = (container: HTMLElement, label: string) =>
  getComputedStyle(container.querySelector<HTMLElement>(`[aria-label="${label}"]`)!);
