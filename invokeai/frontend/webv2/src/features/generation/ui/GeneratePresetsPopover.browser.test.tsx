import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GeneratePresetsPopover } from './GeneratePresetsPopover';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en' });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ADAPTER = {
  generateValues: { getSnapshot: () => ({}), subscribe: () => () => {} },
  models: { catalog: [] },
  notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn() },
  presets: {
    presets: [
      { id: 'preset-1', label: 'Portrait', values: {} },
      { id: 'preset-2', label: 'Landscape', values: {} },
    ],
    remove: vi.fn(),
    rename: vi.fn(),
  },
  project: { activeProjectId: 'project-1' },
  settings: { patchGenerateSettings: vi.fn() },
} as unknown as GenerationUiAdapter;

/** The popover is a dialog too; this selects the modal dialog's content. */
const modalDialog = () => document.querySelector('[data-scope="dialog"][data-part="content"]');
const renameInput = () => modalDialog()?.querySelector<HTMLInputElement>('input[name="renameValue"]');
/** Untranslated in tests, so every row's rename button shares one label; rows keep catalog order. */
const renameButtons = () => document.querySelectorAll<HTMLElement>('[aria-label="widgets.generate.renamePresetNamed"]');

const openRename = async (index: number) => {
  if (renameButtons().length === 0) {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="widgets.generate.presets"]')!));
    await expect.poll(() => renameButtons().length).toBeGreaterThan(index);
  }

  await act(() => renameButtons()[index]!.click());
  await expect.poll(() => modalDialog()?.getAttribute('data-state')).toBe('open');
};

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <GeneratePresetsPopover />
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('GeneratePresetsPopover', () => {
  it('animates the rename dialog out instead of unmounting it on close', async () => {
    await openRename(0);
    expect(renameInput()?.value).toBe('Portrait');

    const frames = closingFrames(
      await recordDialogExit(modalDialog()!, () => act(() => userEvent.keyboard('{Escape}')))
    );
    await expect.poll(() => modalDialog()).toBeNull();

    expect(frames).not.toHaveLength(0);
  });

  it('starts a rename reopened during the exit animation from the newly chosen preset', async () => {
    await openRename(0);
    await act(() => userEvent.type(renameInput()!, ' draft'));
    await act(() => userEvent.keyboard('{Escape}'));
    // Still animating out: a dialog reused for the next open would keep this draft and the old preset's name.
    expect(modalDialog()?.getAttribute('data-state')).toBe('closed');

    await openRename(1);

    expect(renameInput()?.value).toBe('Landscape');
  });
});
