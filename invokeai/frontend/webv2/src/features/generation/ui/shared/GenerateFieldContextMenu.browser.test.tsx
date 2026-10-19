import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GenerateFieldContextMenu } from './GenerateFieldContextMenu';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const FIELDS = ['Steps', 'CFG'].map((name) => ({ copyValue: () => name, name }));
const onReset = vi.fn();

/** Two stacked fields, like Steps over CFG; each menu names its field so the open one is identifiable. */
const renderFields = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        {FIELDS.map(({ copyValue, name }) => (
          <GenerateFieldContextMenu key={name} copyValue={copyValue} resetLabel={`Reset ${name}`} onReset={onReset}>
            <Box data-field={name} h="7" w="40rem">
              {name}
            </Box>
          </GenerateFieldContextMenu>
        ))}
      </ChakraProvider>
    )
  );
};

const field = (name: string) => host!.querySelector<HTMLElement>(`[data-field="${name}"]`)!;
/** Right-clicks near a field's left edge, or past where the other field's menu (opened at its left edge) reaches. */
const rightClick = (element: HTMLElement, x = 8) =>
  act(() => userEvent.click(element, { button: 'right', position: { x, y: 8 } }));
const openMenus = () => [...document.querySelectorAll<HTMLElement>('[role="menu"][data-state="open"]')];
const settle = () =>
  new Promise((resolve) => {
    setTimeout(resolve, 100);
  });

describe('GenerateFieldContextMenu', () => {
  it('moves the menu to a second field right-clicked while the first is open', async () => {
    await renderFields();

    await rightClick(field('Steps'));
    await expect.poll(() => openMenus()[0]?.textContent).toContain('Reset Steps');
    await rightClick(field('CFG'), 500);
    await settle();

    expect(openMenus()).toHaveLength(1);
    expect(openMenus()[0]?.textContent).toContain('Reset CFG');
  });

  it('keeps a menu open when right-clicked itself, then still moves to the next field', async () => {
    await renderFields();

    await rightClick(field('Steps'));
    await expect.poll(() => openMenus()).toHaveLength(1);
    await settle();
    const menu = openMenus()[0]!;

    await rightClick(menu, 20);
    await settle();
    expect(openMenus()).toEqual([menu]);

    await rightClick(field('CFG'), 500);
    await settle();
    expect(openMenus()).toHaveLength(1);
    expect(openMenus()[0]?.textContent).toContain('Reset CFG');
  });
});
