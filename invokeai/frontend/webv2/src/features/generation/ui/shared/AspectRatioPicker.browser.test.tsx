/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop */
import '@fontsource/inter/index.css';
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
  await page.viewport(1280, 720);
});

const cinema: AspectRatioPreset = { height: 1, id: 'cinema', label: 'Cinema', width: 2.39 };
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

describe('AspectRatioPicker choosing', () => {
  it('moves through the ratio grid with the arrow keys, chooses with Enter, then closes back on the trigger', async () => {
    const { onChange } = await render({ lockedRatio: 16 / 9 });

    await openPicker();
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
});

describe('AspectRatioPicker saved ratios', () => {
  it('adds a ratio prefilled from the current size, with Enter, and focuses its new row', async () => {
    const { onChange } = await render({ initial: [] });

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Add ratio' }));

    await expect.element(nameInput()).toHaveFocus();
    await expect.element(widthTerm()).toHaveValue('12');
    await expect.element(heightTerm()).toHaveValue('5');
    await expect.element(widthTerm()).toHaveAccessibleDescription('Picking it sets 1536×640');

    await userEvent.keyboard('Banner{Enter}');

    const row = page.getByRole('button', { name: 'Banner, 12:5, 1536×640' });
    await expect.element(row).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
    expect(isOpen()).toBe(true);
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
    await expect.element(widthTerm()).toBeInvalid();
    await expect.element(heightTerm()).toBeValid();

    await userEvent.type(nameInput(), 'Strip');
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '9');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1');
    await userEvent.keyboard('{Enter}');

    await expect.element(page.getByText('Use a ratio between 1:8 and 8:1.')).toBeVisible();
    await expect.element(page.getByText('Enter a name.')).not.toBeInTheDocument();
    await expect.element(widthTerm()).toHaveFocus();
    await expect.element(widthTerm()).toBeInvalid();
    await expect.element(heightTerm()).toBeInvalid();
    expect(page.getByRole('button', { name: /^Strip,/ }).elements()).toHaveLength(0);

    for (const unreadable of ['1e1', '0x10']) {
      await userEvent.clear(widthTerm());
      await userEvent.type(widthTerm(), unreadable);
      await expect.element(page.getByText('Enter two numbers above 0.')).toBeVisible();
      await expect.element(heightTerm()).toBeValid();
    }

    for (const precise of ['1.234', '1,000']) {
      await userEvent.clear(widthTerm());
      await userEvent.type(widthTerm(), precise);
      await expect.element(page.getByText('Use up to two decimals.')).toBeVisible();
      await expect.element(widthTerm()).toBeInvalid();
      await expect.element(widthTerm()).toHaveAccessibleDescription('Use up to two decimals.');
      await expect.element(heightTerm()).toBeValid();
      await expect.element(heightTerm()).toHaveAccessibleDescription('');
    }

    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '2');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '0.125');
    await userEvent.click(page.getByRole('button', { name: 'Add', exact: true }));
    await expect.element(heightTerm()).toHaveFocus();
    await expect.element(widthTerm()).toBeValid();

    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '0');
    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1.234');
    await expect.element(page.getByText('Enter two numbers above 0.')).toBeVisible();
    await expect.element(widthTerm()).toBeInvalid();
    await expect.element(heightTerm()).toBeValid();
    await expect.element(heightTerm()).toHaveAccessibleDescription('');
    await userEvent.clear(widthTerm());
    await userEvent.type(widthTerm(), '2');
    await expect.element(page.getByText('Use up to two decimals.')).toBeVisible();
    await expect.element(widthTerm()).toBeValid();
    await expect.element(heightTerm()).toBeInvalid();

    await userEvent.clear(heightTerm());
    await userEvent.type(heightTerm(), '1.');

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

  it('deletes a row without asking and moves focus to Add ratio once none are left', async () => {
    const { onChange, removed } = await render();

    await openPicker();
    await userEvent.click(page.getByRole('button', { name: 'Delete ratio Cinema (2.39:1)' }));

    expect(removed).toEqual(['cinema']);
    await expect.element(page.getByRole('button', { name: 'Add ratio' })).toHaveFocus();
    expect(onChange).not.toHaveBeenCalled();
    expect(isOpen()).toBe(true);
  });
});
