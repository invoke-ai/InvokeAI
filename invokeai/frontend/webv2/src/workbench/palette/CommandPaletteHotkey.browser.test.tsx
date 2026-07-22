import type * as settingsStoreModule from '@workbench/settings/store';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const testPreferences = vi.hoisted(() => ({ customHotkeys: {} as Record<string, string[]> }));

vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<typeof settingsStoreModule>()),
  useWorkbenchPreferenceSelector: <Selected,>(selector: (preferences: typeof testPreferences) => Selected): Selected =>
    selector(testPreferences),
  useWorkbenchPreferences: () => testPreferences,
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { hotkey?: string }) =>
      key === 'commandPalette.buttonTooltip'
        ? `Command palette (${options?.hotkey ?? ''})`
        : key === 'commandPalette.buttonLabel'
          ? 'Command palette'
          : key,
  }),
}));

vi.mock('./LaunchpadCommandPaletteDialog', () => ({ default: () => <div data-testid="launchpad-palette" /> }));

import { LaunchpadCommandPalette } from './LaunchpadCommandPalette';
import { PaletteButton } from './PaletteButton';
import { closeCommandPalette, useIsCommandPaletteOpen } from './paletteStore';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const OpenState = () => <output data-testid="palette-state">{useIsCommandPaletteOpen() ? 'open' : 'closed'}</output>;

const renderLaunchpad = async (): Promise<void> => {
  await act(() =>
    root?.render(
      <>
        <input aria-label="Editable target" />
        <LaunchpadCommandPalette />
        <OpenState />
      </>
    )
  );
};

const press = async ({ altKey = false, code, ctrlKey = false, key }: KeyboardEventInit): Promise<void> => {
  const target = document.querySelector<HTMLInputElement>('[aria-label="Editable target"]') ?? window;

  await act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { altKey, bubbles: true, cancelable: true, code, ctrlKey, key }));
  });
};

const expectPaletteState = (state: 'closed' | 'open'): void => {
  expect(document.querySelector('[data-testid="palette-state"]')?.textContent).toBe(state);
};

beforeEach(() => {
  testPreferences.customHotkeys = {};
  closeCommandPalette();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => {
    closeCommandPalette();
    root?.unmount();
  });
  host?.remove();
  host = null;
  root = null;
});

describe('Launchpad command-palette hotkeys', () => {
  it('opens and closes from the default Mod+K binding, including from editable targets', async () => {
    await renderLaunchpad();
    document.querySelector<HTMLInputElement>('[aria-label="Editable target"]')?.focus();

    await press({ code: 'KeyK', ctrlKey: true, key: 'k' });
    expectPaletteState('open');

    await press({ code: 'KeyK', ctrlKey: true, key: 'k' });
    expectPaletteState('closed');
  });

  it('replaces the default binding and installs every custom binding', async () => {
    testPreferences.customHotkeys = { 'app.openCommandPalette': ['alt+p', 'alt+o'] };
    await renderLaunchpad();

    await press({ code: 'KeyK', ctrlKey: true, key: 'k' });
    expectPaletteState('closed');

    await press({ altKey: true, code: 'KeyP', key: 'p' });
    expectPaletteState('open');
    await act(() => closeCommandPalette());

    await press({ altKey: true, code: 'KeyO', key: 'o' });
    expectPaletteState('open');
  });

  it('installs no listener when the command is unbound', async () => {
    testPreferences.customHotkeys = { 'app.openCommandPalette': [] };
    await renderLaunchpad();

    await press({ code: 'KeyK', ctrlKey: true, key: 'k' });
    expectPaletteState('closed');
  });

  it('replaces listeners when preferences change without remounting the Launchpad host', async () => {
    testPreferences.customHotkeys = { 'app.openCommandPalette': ['alt+p'] };
    await renderLaunchpad();

    testPreferences.customHotkeys = { 'app.openCommandPalette': ['alt+o'] };
    await renderLaunchpad();

    await press({ altKey: true, code: 'KeyP', key: 'p' });
    expectPaletteState('closed');

    await press({ altKey: true, code: 'KeyO', key: 'o' });
    expectPaletteState('open');
  });
});

describe('PaletteButton hotkey tooltip', () => {
  const renderButton = async (): Promise<HTMLButtonElement> => {
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <PaletteButton />
        </ChakraProvider>
      )
    );

    const button = document.querySelector<HTMLButtonElement>('button[aria-label="Command palette"]');
    expect(button).not.toBeNull();
    return button!;
  };

  it('displays the first effective custom binding', async () => {
    testPreferences.customHotkeys = { 'app.openCommandPalette': ['alt+p', 'alt+o'] };
    const button = await renderButton();

    await act(async () => {
      await userEvent.hover(button);
      await vi.waitFor(() => expect(document.body.textContent).toContain('Command palette (alt+p)'));
    });
  });

  it('falls back to the plain label when the command is unbound', async () => {
    testPreferences.customHotkeys = { 'app.openCommandPalette': [] };
    const button = await renderButton();

    await act(async () => {
      await userEvent.hover(button);
      await vi.waitFor(() => expect(document.body.textContent).toContain('Command palette'));
    });
    expect(document.body.textContent).not.toContain('Command palette (');
  });
});
