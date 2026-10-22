import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, createActionToast, toaster } from '@platform/ui/toaster';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { DEFAULT_THEME_ID, THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { getContrastRatio, toRgb } from './contrastRatio.testing';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  applyThemeToRoot(DEFAULT_THEME_ID);
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('toast contrast', () => {
  it.each(['error', 'info', 'success', 'warning'] as const)('renders a %s description at AA contrast', async (type) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <AppToaster />
        </ChakraProvider>
      );
    });
    await act(() => {
      toaster.create({
        description: 'Layer thumbnail rasterization failed.',
        duration: Infinity,
        title: 'Error',
        type,
      });
    });
    await expect.poll(() => document.querySelector('[data-part="description"]')).not.toBeNull();
    const description = document.querySelector<HTMLElement>('[data-part="description"]')!;
    const toastRoot = description.closest<HTMLElement>('[data-part="root"]')!;
    const style = getComputedStyle(description);

    const ratio = getContrastRatio(style.color, getComputedStyle(toastRoot).backgroundColor, Number(style.opacity));

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });

  /** sRGB channels and alpha of a computed color: `rgb()`/`rgba()`, or `color(srgb …)` as `color-mix()` computes. */
  const parseColor = (color: string): [number, number, number, number] => {
    const srgb = /^color\(srgb ([\d.]+) ([\d.]+) ([\d.]+)(?: \/ ([\d.]+))?\)$/.exec(color);

    if (srgb) {
      return [Number(srgb[1]) * 255, Number(srgb[2]) * 255, Number(srgb[3]) * 255, Number(srgb[4] ?? 1)];
    }

    const [red, green, blue] = toRgb(color);
    const alpha = /^rgba\([^)]*,\s*([\d.]+)\)$/.exec(color)?.[1];

    return [red, green, blue, alpha === undefined ? 1 : Number(alpha)];
  };

  /** A translucent fill as it reads over the toast. */
  const flatten = (fill: string, under: string): string => {
    const [red, green, blue, alpha] = parseColor(fill);
    const base = toRgb(under);
    const mixed = [red, green, blue].map((channel, index) => Math.round(channel * alpha + base[index]! * (1 - alpha)));

    return `rgb(${mixed.join(' ')})`;
  };

  it.each(['error', 'info', 'success', 'warning'] as const)(
    'keeps %s toast buttons readable at rest, hovered and focused in every theme',
    async (type) => {
      host = document.createElement('div');
      document.body.append(host);
      root = createRoot(host);
      await act(() => {
        root?.render(
          <ChakraProvider value={system}>
            <AppToaster />
          </ChakraProvider>
        );
        createActionToast({
          actions: [
            { label: 'Retry', onClick: () => {} },
            { label: 'Show in Gallery', onClick: () => {} },
          ],
          duration: Infinity,
          title: 'Saved',
          type,
        });
      });
      await expect
        .poll(() => document.querySelectorAll('[data-part="root"][data-scope="toast"] button').length)
        .toBe(3);
      const toastRoot = document.querySelector<HTMLElement>('[data-part="root"][data-scope="toast"]')!;
      const buttons = [...toastRoot.querySelectorAll<HTMLButtonElement>('button')].filter(
        (button) => button.textContent
      );
      // Compare settled palettes, independently of theme and pointer transition durations.
      for (const element of [toastRoot, ...buttons]) {
        element.style.transition = 'none';
      }
      // Resolve each theme's trigger token in the toast's status scope.
      const probe = document.createElement('span');
      probe.style.background = 'var(--toast-trigger-bg)';
      toastRoot.append(probe);

      for (const theme of THEMES) {
        applyThemeToRoot(theme.id);
        const surface = getComputedStyle(toastRoot).backgroundColor;
        for (const button of buttons) {
          expect(
            getContrastRatio(getComputedStyle(button).color, surface, 1),
            `${theme.id} ${button.textContent} at rest`
          ).toBeGreaterThanOrEqual(4.5);
        }
      }

      // Hold each real CSS state while changing themes, rather than repeating pointer setup for every palette.
      for (const button of buttons) {
        await userEvent.hover(button);
        for (const theme of THEMES) {
          applyThemeToRoot(theme.id);
          const surface = getComputedStyle(toastRoot).backgroundColor;
          const triggerFill = getComputedStyle(probe).backgroundColor;
          const label = `${theme.id} ${button.textContent}`;
          await expect
            .poll(() => getComputedStyle(button).backgroundColor, { message: `${label} hover fill` })
            .toBe(triggerFill);
          expect(
            getContrastRatio(getComputedStyle(button).color, flatten(triggerFill, surface), 1),
            `${label} hovered`
          ).toBeGreaterThanOrEqual(4.5);
        }

        await userEvent.unhover(button);
        button.focus();
        await userEvent.keyboard('{Shift>}{/Shift}');
        for (const theme of THEMES) {
          applyThemeToRoot(theme.id);
          const label = `${theme.id} ${button.textContent}`;
          expect(button.matches(':focus-visible'), `${label} keyboard focus`).toBe(true);
          expect(getComputedStyle(button).outlineStyle, `${label} focus ring`).toBe('solid');
          expect(
            getContrastRatio(getComputedStyle(button).outlineColor, getComputedStyle(toastRoot).backgroundColor, 1),
            `${label} focus ring`
          ).toBeGreaterThanOrEqual(3);
        }
        button.blur();
      }
    }
  );
});
