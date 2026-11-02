/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop */
import type {
  AspectRatioPreset,
  AspectRatioPresetDraft,
  AspectRatioPresetsPort,
} from '@features/generation/core/aspectRatioPresets';

import { ChakraProvider } from '@chakra-ui/react';
import { calculateNewSize } from '@features/generation/core/settings';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { AspectRatioPicker } from './AspectRatioPicker';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const cinema: AspectRatioPreset = { height: 1, id: 'cinema', label: 'Cinema', width: 2.39 };
const square: AspectRatioPreset = { height: 1, id: 'square', label: 'Square', width: 1 };

/** An in-memory port: what the account's preference would hold, without the toast the real one adds on delete. */
const Harness = ({
  currentSize,
  initial,
  lockedRatio,
  removed,
  savesNothing,
  onChange,
}: {
  currentSize: { height: number; width: number };
  initial: AspectRatioPreset[];
  lockedRatio: number | null;
  removed: string[];
  /** The list filled up elsewhere (another tab) after the form opened. */
  savesNothing: boolean;
  onChange: (ratio: number | null) => void;
}) => {
  const [presets, setPresets] = useState(initial);
  const port: AspectRatioPresetsPort = {
    isFull: false,
    presets,
    remove: (id) => {
      removed.push(id);
      setPresets((current) => current.filter((preset) => preset.id !== id));
    },
    save: (draft: AspectRatioPresetDraft) => {
      if (savesNothing) {
        return null;
      }
      const preset = { ...draft, id: `saved-${presets.length}` };
      setPresets((current) => [...current, preset]);
      return preset;
    },
    update: (id, draft) =>
      setPresets((current) => current.map((preset) => (preset.id === id ? { ...preset, ...draft } : preset))),
  };

  return (
    <AspectRatioPicker
      currentSize={currentSize}
      getSizeForRatio={(ratio) => calculateNewSize(ratio, currentSize.width * currentSize.height, 8)}
      lockedRatio={lockedRatio}
      presets={port}
      onChange={onChange}
    />
  );
};

const render = async ({
  currentSize = { height: 640, width: 1536 },
  initial = [cinema],
  lockedRatio = null,
  savesNothing = false,
}: {
  currentSize?: { height: number; width: number };
  initial?: AspectRatioPreset[];
  lockedRatio?: number | null;
  savesNothing?: boolean;
} = {}) => {
  host = document.createElement('div');
  host.style.width = '280px';
  document.body.append(host);
  root = createRoot(host);
  const onChange = vi.fn();
  const removed: string[] = [];

  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness
            currentSize={currentSize}
            initial={initial}
            lockedRatio={lockedRatio}
            removed={removed}
            savesNothing={savesNothing}
            onChange={onChange}
          />
        </ChakraProvider>
      </I18nextProvider>
    )
  );

  return { onChange, removed };
};

const trigger = () => page.getByRole('button', { name: /^Aspect ratio/ });
const dialog = () => document.querySelector<HTMLElement>('[data-scope="popover"][data-part="content"]');
const isOpen = () => dialog()?.getAttribute('data-state') === 'open';
const openPicker = async () => {
  await userEvent.click(trigger());
  await expect.poll(isOpen).toBe(true);
};
const nameInput = () => page.getByRole('textbox', { name: 'Ratio name' });
const widthTerm = () => page.getByRole('textbox', { name: 'Ratio width' });
const heightTerm = () => page.getByRole('textbox', { name: 'Ratio height' });

describe('AspectRatioPicker trigger', () => {
  it('names the locked ratio by its saved name, built-in id or terms, and reads Free while unlocked', async () => {
    await render({ lockedRatio: 2.39 });
    await expect.element(trigger()).toHaveAccessibleName('Aspect ratio Cinema 2.39:1');
    await act(() => root!.unmount());

    await render({ lockedRatio: 16 / 9 });
    await expect.element(trigger()).toHaveAccessibleName('Aspect ratio 16:9');
    await act(() => root!.unmount());

    // A locked ratio no saved ratio or built-in names reads as its terms, not as Free.
    await render({ lockedRatio: 2.4 });
    await expect.element(trigger()).toHaveAccessibleName('Aspect ratio 12:5');
    await act(() => root!.unmount());

    await render({ lockedRatio: null });
    await expect.element(trigger()).toHaveAccessibleName('Aspect ratio Free');
  });

  it('prefers a saved ratio over the built-in it equals', async () => {
    await render({ initial: [square], lockedRatio: 1 });

    await expect.element(trigger()).toHaveAccessibleName('Aspect ratio Square 1:1');
  });
});

describe('AspectRatioPicker choosing', () => {
  it('moves through the ratio grid with the arrow keys, chooses with Enter, then closes back on the trigger', async () => {
    const { onChange } = await render({ lockedRatio: 16 / 9 });

    await openPicker();
    // Opening lands on the current ratio.
    await expect.element(page.getByRole('radio', { name: '16:9', exact: true })).toHaveFocus();

    await userEvent.keyboard('{ArrowRight}');
    await expect.element(page.getByRole('radio', { name: '3:2', exact: true })).toHaveFocus();
    // Down a row of five: 3:2 sits above 9:16.
    await userEvent.keyboard('{ArrowDown}');
    await expect.element(page.getByRole('radio', { name: '9:16', exact: true })).toHaveFocus();
    await userEvent.keyboard('{Home}');
    await expect.element(page.getByRole('radio', { name: 'Free', exact: true })).toHaveFocus();
    await userEvent.keyboard('{ArrowRight}{Enter}');

    expect(onChange).toHaveBeenCalledExactlyOnceWith(21 / 9);
    await expect.poll(isOpen).toBe(false);
    await expect.element(trigger()).toHaveFocus();
  });

  it('chooses a saved ratio from its row, which names its ratio and the size it sets', async () => {
    const { onChange } = await render();

    await openPicker();
    // 1536×640 holds 983,040 pixels; at 2.39:1 that is 1532.8×641.3, snapped to 1536×640.
    await userEvent.click(page.getByRole('button', { name: 'Cinema, 2.39:1, 1536×640' }));

    expect(onChange).toHaveBeenCalledExactlyOnceWith(2.39);
    await expect.poll(isOpen).toBe(false);
  });

  it('chooses Free as null and leaves the current ratio alone when chosen again', async () => {
    const { onChange } = await render({ lockedRatio: 1 });

    await openPicker();
    await userEvent.click(page.getByRole('radio', { name: '1:1', exact: true }));
    expect(onChange).not.toHaveBeenCalled();

    await openPicker();
    await userEvent.click(page.getByRole('radio', { name: 'Free', exact: true }));
    expect(onChange).toHaveBeenCalledExactlyOnceWith(null);
  });
});

describe('AspectRatioPicker current choice', () => {
  it('checks the built-in a saved ratio equals, and choosing either only closes', async () => {
    const { onChange } = await render({ initial: [square], lockedRatio: 1 });

    await openPicker();
    await expect.element(page.getByRole('radio', { exact: true, name: '1:1' })).toBeChecked();
    await expect.element(page.getByRole('radio', { exact: true, name: 'Free' })).not.toBeChecked();
    await userEvent.click(page.getByRole('radio', { exact: true, name: '1:1' }));
    await expect.poll(isOpen).toBe(false);

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: /^Square, 1:1, / }));
    await expect.poll(isOpen).toBe(false);
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('AspectRatioPicker saved ratios', () => {
  it('invites a first ratio while none are saved', async () => {
    await render({ initial: [] });
    await openPicker();

    await expect.element(page.getByText('Name a ratio you use often to pick it here at any size.')).toBeVisible();
  });

  it('fits an 18-character name untruncated beside its ratio, size and visible actions', async () => {
    // Current, so the name is bold: its widest form.
    await render({
      initial: [{ height: 1, id: 'insta', label: 'Instagram portrait', width: 2.39 }],
      lockedRatio: 2.39,
    });
    await openPicker();

    const row = page.getByRole('button', { name: 'Instagram portrait, 2.39:1, 1536×640' });
    await row.hover();
    await expect.element(page.getByRole('button', { name: 'Delete ratio Instagram portrait (2.39:1)' })).toBeVisible();
    const name = row.getByText('Instagram portrait', { exact: true }).element() as HTMLElement;

    await expect
      .poll(() => getComputedStyle(page.getByRole('button', { name: /^Delete ratio/ }).element()).opacity)
      .toBe('1');
    expect(name.scrollWidth).toBeLessThanOrEqual(name.clientWidth);
  });

  it('adds a ratio prefilled from the current size, with Enter, and focuses its new row', async () => {
    const { onChange } = await render({ initial: [] });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));

    await expect.element(nameInput()).toHaveFocus();
    await expect.element(widthTerm()).toHaveValue('12');
    await expect.element(heightTerm()).toHaveValue('5');
    await expect.element(page.getByText('From the current size · 1536×640')).toBeVisible();
    await expect.element(widthTerm()).toHaveAccessibleDescription('From the current size · 1536×640');

    await userEvent.keyboard('Banner{Enter}');

    const row = page.getByRole('button', { name: 'Banner, 12:5, 1536×640' });
    await expect.element(row).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
    expect(isOpen()).toBe(true);
  });

  it('prefills a locked ratio, not the size snapped to it, and says where the terms came from', async () => {
    // 1368×768 is exactly 57:32, the 16:9 lock snapped to the grid.
    await render({ currentSize: { height: 768, width: 1368 }, initial: [], lockedRatio: 16 / 9 });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));

    await expect.element(widthTerm()).toHaveValue('16');
    await expect.element(heightTerm()).toHaveValue('9');
    await expect.element(page.getByText('From the locked ratio · 16:9')).toBeVisible();
  });

  it('keeps focus in the popover when a full list saves nothing', async () => {
    await render({ savesNothing: true });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));
    await userEvent.keyboard('Banner{Enter}');

    await expect.element(page.getByRole('button', { name: 'Cinema, 2.39:1, 1536×640' })).toHaveFocus();
  });

  it('previews the ratio as its terms are typed, and a dashed square while they make none', async () => {
    await render();
    await openPicker();
    const shape = (form: string) =>
      page.getByRole('form', { name: form }).element().querySelector<HTMLElement>('[data-shape]')!;
    const aspect = (form: string) => {
      const rect = shape(form).getBoundingClientRect();
      return rect.width / rect.height;
    };

    // Add starts from the current size, 1536×640 (12:5).
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));
    await userEvent.keyboard('Wide');
    expect(aspect('Add ratio')).toBeCloseTo(2.4, 1);
    expect(shape('Add ratio').getBoundingClientRect().width).toBeGreaterThan(60);
    // Its framed well spans the name field and the terms row beside it.
    const well = page.getByRole('form', { name: 'Add ratio' }).element().querySelector('[data-preview-well]')!;
    const wellRect = well.getBoundingClientRect();
    expect(wellRect.top).toBeLessThanOrEqual((nameInput().element() as HTMLElement).getBoundingClientRect().top);
    expect(wellRect.bottom).toBeGreaterThanOrEqual(
      (widthTerm().element() as HTMLElement).getBoundingClientRect().bottom
    );
    // A square well.
    expect(wellRect.width).toBeCloseTo(wellRect.height, 0);
    await expect.element(nameInput()).toHaveAttribute('placeholder', 'Ratio name');

    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '4');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '5');
    expect(aspect('Add ratio')).toBeCloseTo(0.8, 1);

    // A comma decimal reads as a decimal.
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '2,39');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1');
    expect(aspect('Add ratio')).toBeCloseTo(2.39, 1);

    const formHeight = page.getByRole('form', { name: 'Add ratio' }).element().getBoundingClientRect().height;
    for (const invalid of ['0', '9']) {
      await userEvent.clear(widthTerm());
      await userEvent.type(widthTerm(), invalid);
      expect(shape('Add ratio').hasAttribute('data-placeholder')).toBe(true);
      expect(getComputedStyle(shape('Add ratio')).borderStyle).toBe('dashed');
      expect(aspect('Add ratio')).toBeCloseTo(1, 2);
    }
    // Neither the shape nor the terms error on submit changes the form's height.
    await userEvent.click(page.getByRole('button', { name: 'Add', exact: true }));
    await expect.element(page.getByText('Use a ratio between 1:8 and 8:1.')).toBeVisible();
    expect(page.getByRole('form', { name: 'Add ratio' }).element().getBoundingClientRect().height).toBe(formHeight);

    // Edit starts from the saved ratio.
    await userEvent.keyboard('{Escape}');
    await userEvent.click(page.getByRole('button', { name: 'Edit ratio Cinema (2.39:1)' }));
    expect(aspect('Edit ratio')).toBeCloseTo(2.39, 1);
  });

  it('keeps the form open with inline errors for a missing name, a zero term and a ratio past 8:1', async () => {
    await render({ initial: [] });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '0');
    await userEvent.click(page.getByRole('button', { name: 'Add', exact: true }));

    await expect.element(page.getByText('Enter a name.')).toBeVisible();
    await expect.element(page.getByText('Enter two numbers above 0.')).toBeVisible();
    await expect.element(nameInput()).toHaveFocus();

    await userEvent.type(nameInput(), 'Strip');
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '9');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1');
    await userEvent.keyboard('{Enter}');

    await expect.element(page.getByText('Use a ratio between 1:8 and 8:1.')).toBeVisible();
    await expect.element(page.getByText('Enter a name.')).not.toBeInTheDocument();
    await expect.element(widthTerm()).toHaveFocus();
    expect(page.getByRole('button', { name: /^Strip,/ }).elements()).toHaveLength(0);

    // Only plain decimals read as terms, never what `Number` would also accept.
    for (const unreadable of ['1e1', '0x10', '1,000']) {
      await userEvent.clear(widthTerm());
      await userEvent.type(widthTerm(), unreadable);
      await expect.element(page.getByText('Enter two numbers above 0.')).toBeVisible();
    }

    // A comma decimal reads as a decimal.
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '2,5');
    await userEvent.keyboard('{Enter}');

    await expect.element(page.getByRole('button', { name: 'Strip, 2.5:1, 1568×624' })).toHaveFocus();
  });

  it('edits a row in place and saves the new name and ratio', async () => {
    await render();

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Edit ratio Cinema (2.39:1)' }));
    await expect.element(nameInput()).toHaveValue('Cinema');
    await expect.element(widthTerm()).toHaveValue('2.39');
    // The terms come from the saved ratio, so the hint names the size they set, as they change.
    await expect.element(page.getByText('Picking it sets 1536×640')).toBeVisible();
    await expect.element(page.getByText(/^From the current size/)).not.toBeInTheDocument();
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '4');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '5');
    // 1536×640 at 4:5 is 886.8×1108.5, snapped to 888×1112.
    await expect.element(page.getByText('Picking it sets 888×1112')).toBeVisible();
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1');

    await userEvent.clear(nameInput());
    await userEvent.type(nameInput(), 'Scope');
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '2.4');
    await userEvent.click(page.getByRole('button', { name: 'Save', exact: true }));

    await expect.element(page.getByRole('button', { name: 'Scope, 2.4:1, 1536×640' })).toHaveFocus();
  });

  it('moves focus to the next row Delete after a delete', async () => {
    await render({ initial: [cinema, square] });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Delete ratio Cinema (2.39:1)' }));

    await expect.element(page.getByRole('button', { name: 'Delete ratio Square (1:1)' })).toHaveFocus();
  });

  it('deletes a row without asking and moves focus to Add ratio once none are left', async () => {
    const { onChange, removed } = await render();

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Delete ratio Cinema (2.39:1)' }));

    expect(removed).toEqual(['cinema']);
    await expect.element(page.getByRole('button', { name: 'Add ratio' })).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
    expect(isOpen()).toBe(true);
  });

  it('cancels the form with Escape, keeping the popover open, and returns focus to the button that opened it', async () => {
    await render();

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Edit ratio Cinema (2.39:1)' }));
    await expect.element(nameInput()).toHaveFocus();
    await userEvent.keyboard('{Escape}');

    expect(isOpen()).toBe(true);
    await expect.element(page.getByRole('button', { name: 'Edit ratio Cinema (2.39:1)' })).toHaveFocus();

    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));
    await userEvent.keyboard('Discarded{Escape}');
    await expect.element(page.getByRole('button', { name: 'Add ratio' })).toHaveFocus();
    expect(page.getByRole('button', { name: /^Discarded,/ }).elements()).toHaveLength(0);

    // With no form open, Escape closes the popover.
    await userEvent.keyboard('{Escape}');
    await expect.poll(isOpen).toBe(false);
    await expect.element(trigger()).toHaveFocus();
  });
});
