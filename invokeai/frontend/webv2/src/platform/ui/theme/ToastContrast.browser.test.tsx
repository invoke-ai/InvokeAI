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
      toaster.create({ description: 'Layer thumbnail rasterization failed.', title: 'Error', type });
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

  it.each(
    THEMES.flatMap((theme) => (['error', 'info', 'success', 'warning'] as const).map((type) => [theme.id, type]))
  )(
    'keeps %s %s toast buttons readable at rest, hovered and focused, with fills from the toast itself',
    async (themeId, type) => {
      applyThemeToRoot(themeId);
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
        createActionToast({
          actions: [
            { label: 'Retry', onClick: () => {} },
            { label: 'Show in Gallery', onClick: () => {} },
          ],
          title: 'Saved',
          type,
        });
      });
      await expect
        .poll(() => document.querySelectorAll('[data-part="root"][data-scope="toast"] button').length)
        .toBe(3);
      const toastRoot = document.querySelector<HTMLElement>('[data-part="root"][data-scope="toast"]')!;
      const surface = getComputedStyle(toastRoot).backgroundColor;
      // The toast's own trigger fill, resolved the way the buttons resolve it.
      const probe = document.createElement('span');
      probe.style.background = 'var(--toast-trigger-bg)';
      toastRoot.append(probe);
      const triggerFill = getComputedStyle(probe).backgroundColor;
      probe.remove();
      const buttons = [...toastRoot.querySelectorAll<HTMLButtonElement>('button')].filter(
        (button) => button.textContent
      );

      for (const button of buttons) {
        const label = button.textContent;
        const text = getComputedStyle(button).color;

        expect(getContrastRatio(text, surface, 1), `${label} at rest`).toBeGreaterThanOrEqual(4.5);

        await userEvent.hover(button);
        await expect
          .poll(() => getComputedStyle(button).backgroundColor, { message: `${label} hover fill` })
          .toBe(triggerFill);
        expect(getContrastRatio(text, flatten(triggerFill, surface), 1), `${label} hovered`).toBeGreaterThanOrEqual(
          4.5
        );

        await userEvent.unhover(button);
        button.focus();
        await userEvent.keyboard('{Shift>}{/Shift}');
        expect(getComputedStyle(button).outlineStyle, `${label} focus ring`).toBe('solid');
        expect(
          getContrastRatio(getComputedStyle(button).outlineColor, surface, 1),
          `${label} focus ring`
        ).toBeGreaterThanOrEqual(3);
      }
    }
  );
});
