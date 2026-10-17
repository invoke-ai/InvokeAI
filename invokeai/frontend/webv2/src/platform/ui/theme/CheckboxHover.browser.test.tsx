import { Box, Checkbox, ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { compositeColors, getContrastRatio } from '@theme/contrastRatio.testing';
import { system } from '@theme/system';
import { THEMES } from '@theme/themes';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

/**
 * Hovering anywhere in a checkbox's target (label, control, or the invisible outset around the control) must read as
 * a stronger boundary or a shifted fill in every theme, while a disabled checkbox stays as it rests.
 */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  // Read settled colors: reduced motion shortens the hover transition to a millisecond.
  document.documentElement.dataset.reduceMotion = 'true';
});

afterEach(async () => {
  await userEvent.unhover(document.body);
  await act(() => root?.unmount());
  host?.remove();
  delete document.documentElement.dataset.highContrast;
  delete document.documentElement.dataset.reduceMotion;
  applyThemeToRoot('classic');
  host = null;
  root = null;
});

const SAMPLES = [
  { id: 'unchecked', checked: false, disabled: false, invalid: false },
  { id: 'checked', checked: true, disabled: false, invalid: false },
  { id: 'disabled', checked: false, disabled: true, invalid: false },
  { id: 'invalid', checked: false, disabled: false, invalid: true },
  { id: 'invalid-checked', checked: true, disabled: false, invalid: true },
  { id: 'invalid-indeterminate', checked: 'indeterminate', disabled: false, invalid: true },
] as const;

const renderCheckboxes = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Box bg="bg.panel" display="flex" flexDirection="column" gap="4" p="4">
          {SAMPLES.map(({ checked, disabled, id, invalid }) => (
            <Checkbox.Root
              key={id}
              checked={checked}
              colorPalette="accent"
              data-testid={id}
              disabled={disabled}
              invalid={invalid}
              size="sm"
            >
              <Checkbox.HiddenInput />
              <Checkbox.Control />
              <Checkbox.Label>{id}</Checkbox.Label>
            </Checkbox.Root>
          ))}
        </Box>
      </ChakraProvider>
    );
  });

  const panel = getComputedStyle(host.firstElementChild!).backgroundColor;
  const part = (id: string, name: string) =>
    host!.querySelector<HTMLElement>(`[data-testid="${id}"] [data-part="${name}"]`)!;
  const settle = () =>
    new Promise<void>((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    });
  // A point in the outset: left of the control's visible box, level with its middle.
  const hoverOutset = async (id: string) => {
    const label = host!.querySelector<HTMLElement>(`[data-testid="${id}"]`)!;
    const labelRect = label.getBoundingClientRect();
    const control = part(id, 'control').getBoundingClientRect();

    await userEvent.hover(label, {
      position: { x: control.left - labelRect.left - 4, y: control.top - labelRect.top + control.height / 2 },
    });
    await settle();
  };
  const hoverLabel = async (id: string) => {
    await userEvent.hover(part(id, 'label'));
    await settle();
  };
  const read = (id: string) => {
    const style = getComputedStyle(part(id, 'control'));
    const fill = compositeColors(style.backgroundColor, panel);

    return {
      border: getContrastRatio(style.borderTopColor, panel, 1),
      fill,
      mark: getContrastRatio(style.color, fill, 1),
    };
  };

  return { hoverLabel, hoverOutset, read };
};

describe('checkbox hover', () => {
  for (const theme of THEMES) {
    for (const highContrast of [false, true]) {
      it(`strengthens the boundary and shifts the fill on ${theme.id}${highContrast ? ' (high contrast)' : ''}`, async () => {
        applyThemeToRoot(theme.id);
        if (highContrast) {
          document.documentElement.dataset.highContrast = 'true';
        }
        const { hoverOutset, read } = await renderCheckboxes();
        const unchecked = read('unchecked');
        const checked = read('checked');

        await hoverOutset('unchecked');
        expect(read('unchecked').border).toBeGreaterThan(unchecked.border * 1.1);

        await hoverOutset('checked');
        const hoveredChecked = read('checked');

        expect(hoveredChecked.fill).not.toBe(checked.fill);
        // The check stays a legible graphic on the hovered fill.
        expect(hoveredChecked.mark).toBeGreaterThanOrEqual(3);
      });
    }
  }

  it('shows the hover from the label, not on a disabled checkbox', async () => {
    const { hoverLabel, hoverOutset, read } = await renderCheckboxes();
    const unchecked = read('unchecked');
    const disabled = read('disabled');

    await hoverLabel('unchecked');
    expect(read('unchecked').border).toBeGreaterThan(unchecked.border);

    await hoverOutset('disabled');
    expect(read('disabled')).toEqual(disabled);
    await hoverLabel('disabled');
    expect(read('disabled')).toEqual(disabled);
  });

  it.each(['invalid', 'invalid-checked', 'invalid-indeterminate'])(
    'preserves the error appearance of %s when hovering the label or outset',
    async (id) => {
      const { hoverLabel, hoverOutset, read } = await renderCheckboxes();
      const invalid = read(id);

      await hoverLabel(id);
      expect(read(id)).toEqual(invalid);
      await hoverOutset(id);
      expect(read(id)).toEqual(invalid);
    }
  );
});
