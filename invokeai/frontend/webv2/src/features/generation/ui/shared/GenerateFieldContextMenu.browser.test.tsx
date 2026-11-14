import { Box, ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
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

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: { widgets: { generate: { resetFieldNamed: '{{reset}}: {{field}}' } } } } },
});

const FIELDS = ['Steps', 'CFG'].map((name) => ({ copyValue: () => name, name }));
const onReset = vi.fn();

/** Two stacked fields, like Steps over CFG; each menu names its field so the open one is identifiable. */
const renderFields = async (isAtDefault = false) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <button data-testid="before" type="button">
            Before
          </button>
          {FIELDS.map(({ copyValue, name }) => (
            <GenerateFieldContextMenu
              key={name}
              copyValue={copyValue}
              fieldLabel={name}
              isAtDefault={isAtDefault}
              resetLabel={`Reset ${name}`}
              onReset={onReset}
            >
              <Box as="button" data-field={name} h="7" w="40rem">
                {name}
              </Box>
            </GenerateFieldContextMenu>
          ))}
        </I18nextProvider>
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

  it('opens at the pointer with nothing to point at: no arrow, and no arrow-sized gap below the click', async () => {
    await renderFields();
    const steps = field('Steps');

    await rightClick(steps);
    await expect.poll(() => openMenus()[0]?.textContent).toContain('Reset Steps');
    await settle();
    const menu = openMenus()[0]!;

    const arrow = menu.parentElement!.querySelector<HTMLElement>('[data-part="arrow"]');
    expect(arrow?.querySelector<HTMLElement>('[data-part="arrow-tip"]')?.offsetWidth ?? 0).toBe(0);
    // The click lands 8px into the field; the menu opens the default 8px gutter below it, where an arrow would add 6.
    const gap = menu.getBoundingClientRect().top - (steps.getBoundingClientRect().top + 8);
    expect(gap).toBeGreaterThanOrEqual(7);
    expect(gap).toBeLessThan(11);
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

  it('marks a changed field with a dot named for it, reached by Tab first, that resets and refocuses the field', async () => {
    onReset.mockClear();
    await renderFields();
    const dot = host!.querySelector<HTMLButtonElement>('button[aria-label="Reset Steps: Steps"]')!;
    const rect = dot.getBoundingClientRect();

    expect(rect.width).toBeGreaterThanOrEqual(16);
    expect(rect.height).toBeGreaterThanOrEqual(16);

    host!.querySelector<HTMLElement>('[data-testid="before"]')!.focus();
    await act(() => userEvent.keyboard('{Tab}'));
    expect(document.activeElement).toBe(dot);

    await act(() => userEvent.keyboard('{Enter}'));
    expect(onReset).toHaveBeenCalledOnce();
    await expect.poll(() => document.activeElement).toBe(field('Steps'));

    await act(() => root?.unmount());
    host?.remove();
    await renderFields(true);
    expect(host!.querySelector('[data-modified-dot]')).toBeNull();
  });

  it('opens from the keyboard menu key under the focused field and returns focus on close', async () => {
    await renderFields();
    const cfg = field('CFG');
    cfg.focus();

    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
    await expect.poll(() => openMenus()[0]?.textContent).toContain('Reset CFG');
    await settle();
    const menuRect = openMenus()[0]!.getBoundingClientRect();
    const fieldRect = cfg.getBoundingClientRect();

    expect(Math.abs(menuRect.left - fieldRect.left)).toBeLessThan(24);
    expect(menuRect.top).toBeGreaterThanOrEqual(fieldRect.bottom - 2);

    await act(() => userEvent.keyboard('{Escape}'));
    await expect.poll(() => openMenus()).toHaveLength(0);
    expect(document.activeElement).toBe(cfg);
  });

  it('keeps the changed dot inside the 8px gutter at rest, on hover and with its focus ring', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <I18nextProvider i18n={i18n}>
            {/* The panel and its 8px field inset, as a section lays fields out. */}
            <Box data-testid="panel" overflow="hidden" px="2" w="30rem">
              <GenerateFieldContextMenu copyValue={FIELDS[0]!.copyValue} fieldLabel="Steps" onReset={onReset}>
                <Box as="button" borderWidth="1px" data-field="Steps" h="7" w="full">
                  Steps
                </Box>
              </GenerateFieldContextMenu>
            </Box>
          </I18nextProvider>
        </ChakraProvider>
      )
    );
    const panel = host.querySelector<HTMLElement>('[data-testid="panel"]')!.getBoundingClientRect();
    const fieldLeft = field('Steps').getBoundingClientRect().left;
    const button = host.querySelector<HTMLElement>('[data-modified-dot]')!;
    const dot = () => button.querySelector<HTMLElement>('[data-part="dot"]')!.getBoundingClientRect();
    /** The dot's extent, with `ring` px of focus ring around it, must clear the panel edge and the field's border. */
    const expectInGutter = (ring: number) => {
      const rect = dot();
      expect(rect.left - ring).toBeGreaterThanOrEqual(panel.left);
      expect(rect.right + ring).toBeLessThanOrEqual(fieldLeft);
    };

    expect(fieldLeft - panel.left).toBe(8);
    expect(dot().width).toBe(4);
    expectInGutter(1);

    await userEvent.hover(button);
    await expect.poll(() => Math.round(dot().width)).toBe(6);
    expectInGutter(0);

    await userEvent.unhover(button);
    // From the keyboard: the ring is drawn on the dot itself, not the 16px button.
    host.querySelector<HTMLElement>('[data-testid="panel"]')!.focus();
    await userEvent.keyboard('{Tab}');
    expect(document.activeElement).toBe(button);
    await expect.poll(() => Math.round(dot().width)).toBe(6);
    // A real outline, which stays in forced colors, 1px around the dot and still inside the gutter.
    const dotStyle = getComputedStyle(button.querySelector<HTMLElement>('[data-part="dot"]')!);
    expect([dotStyle.outlineStyle, dotStyle.outlineWidth, dotStyle.outlineOffset]).toEqual(['solid', '1px', '0px']);
    expectInGutter(1);
  });
});
