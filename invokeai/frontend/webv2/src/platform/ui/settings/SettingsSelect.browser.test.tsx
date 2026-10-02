import { ChakraProvider, Dialog, Portal } from '@chakra-ui/react';
import { WORKBENCH_LANGUAGE_OPTIONS } from '@platform/i18n/languages';
import { Scrollable } from '@platform/ui/Scrollable';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { SettingDefinition } from './contracts';

import { SettingControl } from './SettingControl';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

const LANGUAGE_FIELD: SettingDefinition = {
  id: 'language',
  kind: 'select',
  label: 'Language',
  options: WORKBENCH_LANGUAGE_OPTIONS,
  scope: 'preference',
};

const SettingsDialog = () => {
  const [value, setValue] = useState<boolean | number | string>('en');
  const [open, setOpen] = useState(true);
  const handleOpenChange = useCallback(({ open: next }: { open: boolean }) => setOpen(next), []);
  return (
    <ChakraProvider value={system}>
      <button type="button">Outside dialog</button>
      <Dialog.Root open={open} onOpenChange={handleOpenChange}>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content h="10rem" w="26rem" overflow="hidden">
              <Dialog.Header>
                <Dialog.Title>Settings</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Scrollable h="full">
                  <SettingControl field={LANGUAGE_FIELD} surface="dialog" value={value} onChange={setValue} />
                  <output>{value}</output>
                </Scrollable>
              </Dialog.Body>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    </ChakraProvider>
  );
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const dialogContent = () => document.querySelector<HTMLElement>('[data-scope="dialog"][data-part="content"]');
const listbox = () => document.querySelector<HTMLElement>('[data-scope="select"][data-part="content"]');
const scrollViewport = () => listbox()?.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]');
const language = () => page.getByRole('combobox', { name: 'Language', exact: true });

beforeEach(async () => {
  await page.viewport(900, 420);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => root.render(<SettingsDialog />));
  await expect.element(language()).toBeVisible();
  await expect.poll(() => dialogContent()?.getAnimations().length).toBe(0);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('settings language select', () => {
  it('escapes a clipped dialog and bounds long options in a ScrollArea', async () => {
    await act(() => language().click());
    await expect.element(page.getByRole('listbox', { name: 'Language', exact: true })).toBeVisible();
    await expect.poll(() => scrollViewport()?.clientHeight).toBeGreaterThan(0);
    const viewport = scrollViewport()!;
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    expect(viewport.clientHeight).toBeLessThanOrEqual(288);
    const dialogRect = dialogContent()!.getBoundingClientRect();
    const menuRect = listbox()!.getBoundingClientRect();
    expect(menuRect.top >= 0 && menuRect.bottom <= window.innerHeight).toBe(true);
    expect(menuRect.top < dialogRect.top || menuRect.bottom > dialogRect.bottom).toBe(true);
    const visiblePoint = document.elementFromPoint(menuRect.x + menuRect.width / 2, menuRect.bottom - 12);
    expect(listbox()?.contains(visiblePoint)).toBe(true);
  });

  it('scrolls to and selects the last language with the keyboard without dismissing the dialog', async () => {
    await act(() => language().click());
    await expect.element(page.getByRole('listbox', { name: 'Language', exact: true })).toBeVisible();
    await act(() => userEvent.keyboard('{End}'));
    await expect.poll(() => scrollViewport()?.scrollTop).toBeGreaterThan(0);
    await expect.element(page.getByRole('option', { name: '漢語', exact: true })).toBeVisible();
    const lastOption = listbox()!.querySelector<HTMLElement>('[role="option"][data-value="zh-Hant"]')!;
    expect(lastOption.getBoundingClientRect().bottom).toBeLessThanOrEqual(
      scrollViewport()!.getBoundingClientRect().bottom + 1
    );
    await act(() => userEvent.keyboard('{Enter}'));
    await expect.element(page.getByRole('dialog', { name: 'Settings', exact: true })).toBeVisible();
    await expect.poll(() => dialogContent()?.querySelector('output')?.textContent).toBe('zh-Hant');
    await expect.element(language()).toHaveFocus();
    // Wait for exit focus restoration before pressing Tab.
    await expect.poll(() => listbox()?.checkVisibility() ?? false).toBe(false);
    await act(() => userEvent.keyboard('{Tab}'));
    await expect.poll(() => dialogContent()?.contains(document.activeElement) ?? false).toBe(true);
  });

  it('dismisses only the language list with Escape and retains dialog focus', async () => {
    await act(() => language().click());
    await expect.element(page.getByRole('listbox', { name: 'Language', exact: true })).toBeVisible();
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => listbox()?.checkVisibility() ?? false).toBe(false);
    await expect.element(page.getByRole('dialog', { name: 'Settings', exact: true })).toBeVisible();
    await expect.element(language()).toHaveFocus();
  });
});
