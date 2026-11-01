import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type { SettingDefinition, SettingLayer } from './contracts';

import { SettingControl } from './SettingControl';

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: translations } } });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FIELD: SettingDefinition = { id: 'showGrid', kind: 'boolean', label: 'Show grid', scope: 'project' };
const noop = () => {};
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
