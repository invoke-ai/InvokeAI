import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { SettingDefinition, SettingLayer } from './contracts';

import { SettingControl } from './SettingControl';

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: translations } } });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FIELD: SettingDefinition = { id: 'showGrid', kind: 'boolean', label: 'Show grid', scope: 'project' };
const SELECT_FIELD: SettingDefinition = {
  id: 'gridSize',
  kind: 'select',
  label: 'Grid size',
  options: [
    { label: 'Small', value: 'small' },
    { label: 'Large', value: 'large' },
  ],
  scope: 'project',
};
const noop = () => {};
const CHANGED_LAYER: SettingLayer = { onResetToBuiltIn: noop, state: 'default' };
const LONG_SELECT_FIELD: SettingDefinition = {
  ...SELECT_FIELD,
  options: [{ label: 'A deliberately long option label that cannot fit beside its setting', value: 'long' }],
};
let host: HTMLDivElement;
let root: Root;

const render = (layer: SettingLayer, onChange: (value: boolean | number | string) => void = noop) =>
  act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <SettingControl field={FIELD} surface="dialog" value layer={layer} onChange={onChange} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('setting layer indicator', () => {
  it('names the setting a pinned value belongs to', async () => {
    await render({
      defaultValueLabel: 'Invoke default: Off',
      onFollowDefault: noop,
      onUseAsDefault: noop,
      state: 'pinned',
    });
    // `aria-haspopup` announces the menu; the name says whose dot it is.
    await expect
      .element(page.getByRole('button', { name: 'Show grid: pinned for this project', exact: true }))
      .toHaveAttribute('aria-haspopup', 'menu');
  });

  it('names the setting a changed default belongs to', async () => {
    await render({ onResetToBuiltIn: noop, state: 'default' });
    await expect
      .element(page.getByRole('button', { name: 'Show grid: changed from Invoke default', exact: true }))
      .toHaveAttribute('aria-haspopup', 'menu');
  });

  it('marks a change it offers no action for with a named glyph instead of a button', async () => {
    await render({ state: 'default' });
    await expect
      .element(page.getByRole('img', { name: 'Show grid: changed from Invoke default', exact: true }))
      .toBeVisible();
    expect(page.getByRole('button').elements()).toHaveLength(0);
  });

  it("runs a menu item inside a boolean row's label without toggling the switch", async () => {
    const onChange = vi.fn();
    const onFollowDefault = vi.fn();
    await render(
      { defaultValueLabel: 'Invoke default: Off', onFollowDefault, onUseAsDefault: noop, state: 'pinned' },
      onChange
    );
    await act(() => page.getByRole('button', { name: 'Show grid: pinned for this project', exact: true }).click());
    const follow = page.getByRole('menuitem', { name: 'Follow my default', exact: true });
    // The hint describes the item; its name stays the label.
    await expect.element(follow).toHaveAccessibleDescription('Invoke default: Off');
    await act(() => follow.click());

    expect(onFollowDefault).toHaveBeenCalledOnce();
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe('quick settings select', () => {
  const renderQuick = (value: string, layer: SettingLayer, onChange: (value: boolean | number | string) => void) =>
    act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <SettingControl field={SELECT_FIELD} surface="quick" value={value} layer={layer} onChange={onChange} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  const select = () => page.getByRole('combobox', { name: 'Grid size', exact: true });

  beforeEach(() => {
    host.style.width = '280px';
  });

  it('leads with its label and dot like a switch row, trailing a bordered select sized to its value', async () => {
    await renderQuick('small', CHANGED_LAYER, noop);

    const trigger = select().element();
    const label = host.querySelector<HTMLElement>('label')!;
    const dot = page.getByRole('button', { name: 'Grid size: changed from Invoke default', exact: true }).element();
    expect(label.textContent).toBe('Grid size');
    expect(trigger.textContent).toBe('Small');
    expect(label.getBoundingClientRect().right).toBeLessThanOrEqual(dot.getBoundingClientRect().left);
    expect(dot.getBoundingClientRect().right).toBeLessThanOrEqual(trigger.getBoundingClientRect().left);
    // At the row's end and only as wide as its value and chevron.
    expect(Math.round(trigger.getBoundingClientRect().right)).toBe(Math.round(host.getBoundingClientRect().right));
    expect(trigger.getBoundingClientRect().width).toBeLessThan(host.getBoundingClientRect().width / 2);
    // The same bordered box as other selects.
    expect(getComputedStyle(trigger).borderTopWidth).toBe('1px');
    expect(getComputedStyle(trigger).borderTopStyle).toBe('solid');
  });

  it('focuses the select from a click on its label and commits a choice from a menu aligned to its end', async () => {
    const onChange = vi.fn();
    await renderQuick('small', CHANGED_LAYER, onChange);

    await userEvent.click(page.elementLocator(host.querySelector('label')!));
    await expect.element(select()).toHaveFocus();
    await userEvent.click(select());
    const listbox = page.getByRole('listbox', { name: 'Grid size', exact: true });
    await expect.element(listbox).toBeVisible();
    const menu = listbox.element().getBoundingClientRect();
    // Wider than the trigger it opens from, and flush with its end.
    expect(menu.width).toBeGreaterThanOrEqual(160);
    expect(Math.abs(menu.right - select().element().getBoundingClientRect().right)).toBeLessThanOrEqual(1);
    await userEvent.click(page.getByRole('option', { name: 'Large', exact: true }));
    expect(onChange).toHaveBeenCalledWith('large');
  });

  it('ellipsizes a long value within 60% of the row', async () => {
    await act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <SettingControl field={LONG_SELECT_FIELD} surface="quick" value="long" onChange={noop} />
          </ChakraProvider>
        </I18nextProvider>
      )
    );

    const trigger = select().element();
    const valueText = trigger.querySelector<HTMLElement>('[data-part="value-text"]')!;
    expect(trigger.getBoundingClientRect().width).toBeLessThanOrEqual(host.getBoundingClientRect().width * 0.6 + 1);
    expect(valueText.scrollWidth).toBeGreaterThan(valueText.clientWidth);
    expect(getComputedStyle(valueText).textOverflow).toBe('ellipsis');
    const chevron = trigger.querySelector<HTMLElement>('[data-part="indicator"]')!;
    expect(valueText.getBoundingClientRect().right).toBeLessThanOrEqual(chevron.getBoundingClientRect().left);
  });

  it('keeps the dot beside the label working', async () => {
    const onResetToBuiltIn = vi.fn();
    await renderQuick('small', { onResetToBuiltIn, state: 'default' }, noop);

    await userEvent.click(page.getByRole('button', { name: 'Grid size: changed from Invoke default', exact: true }));
    await userEvent.click(page.getByRole('menuitem', { name: 'Reset to Invoke default', exact: true }));
    expect(onResetToBuiltIn).toHaveBeenCalledOnce();
  });
});
