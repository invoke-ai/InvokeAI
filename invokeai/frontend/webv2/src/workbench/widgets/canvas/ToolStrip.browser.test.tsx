import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import i18n from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { recordSelectFamilyTool } from './toolFamilyStore';
import { ToolStrip } from './ToolStrip';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type StripEngine = ComponentProps<typeof ToolStrip>['engine'];

/** A live interaction-store double: real get/set/subscribe so the strip re-renders on changes. */
const createFakeEngine = () => {
  const values = new Map<string, unknown>([
    ['activeTool', 'view'],
    ['shapeOptions', { fillEnabled: true, kind: 'rect', strokeEnabled: false, strokeWidth: 8 }],
  ]);
  const listeners = new Map<string, Set<() => void>>();
  const interaction = {
    get: (key: string) => values.get(key),
    getLayerThumbnailStatus: () => 'idle' as const,
    getLayerThumbnailVersion: () => 0,
    set: (key: string, value: unknown) => {
      values.set(key, value);
      listeners.get(key)?.forEach((listener) => listener());
    },
    subscribe: (key: string, listener: () => void) => {
      const bucket = listeners.get(key) ?? new Set<() => void>();
      listeners.set(key, bucket);
      bucket.add(listener);
      return () => bucket.delete(listener);
    },
    subscribeLayerThumbnailStatus: () => () => {},
    subscribeLayerThumbnailVersion: () => () => {},
  };
  const engine = {
    interaction,
    tools: { setTool: (toolId: string) => interaction.set('activeTool', toolId) },
  };
  return {
    engine: engine as unknown as StripEngine,
    get activeTool() {
      return values.get('activeTool');
    },
    get shapeKind() {
      return (values.get('shapeOptions') as { kind: string }).kind;
    },
  };
};

const pointerAt = (type: string, x: number, y: number): PointerEvent =>
  new PointerEvent(type, { bubbles: true, button: 0, clientX: x, clientY: y, pointerId: 1 });

const centre = (element: Element): { x: number; y: number } => {
  const rect = element.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
};

const wait = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

describe('tool strip family slots', () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  beforeAll(async () => {
    const translation = await fetch('/locales/en.json').then((response) => response.json());
    await i18n.use(initReactI18next).init({
      fallbackLng: 'en',
      initAsync: false,
      interpolation: { escapeValue: false },
      lng: 'en',
      resources: { en: { translation } },
    });
  });

  beforeEach(() => {
    recordSelectFamilyTool('marquee');
    localStorage.removeItem('invokeai:v7:webv2:select-family-tool');
  });

  afterEach(async () => {
    await act(() => root?.unmount());
    container?.remove();
    container = null;
    root = null;
  });

  const renderStrip = async (props: Partial<ComponentProps<typeof ToolStrip>> = {}) => {
    const fake = createFakeEngine();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <ToolStrip engine={fake.engine} {...props} />
        </ChakraProvider>
      );
    });
    return fake;
  };

  const button = (name: string) => page.getByRole('button', { name, exact: true });
  const menu = () => page.getByRole('menu');
  const entry = (name: string) => page.getByRole('menuitemradio', { name });
  const openByContextMenu = async (slot: HTMLElement) => {
    await act(() => {
      slot.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.element(menu()).toBeVisible();
    // The menu takes focus once it has registered its dismiss handlers.
    await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('menu');
  };

  it('selects the current subtool on a plain click and shows no menu', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await act(() => userEvent.click(shape));
    expect(fake.activeTool).toBe('shape');
    expect(fake.shapeKind).toBe('rect');
    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(shape.getAttribute('aria-expanded')).toBe('false');
  });

  it('selects the current subtool from the keyboard without opening the menu', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await act(async () => {
      shape.focus();
      await userEvent.keyboard('{Enter}');
    });
    expect(fake.activeTool).toBe('shape');
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('opens on hold, highlights the entry the drag is over, and selects the entry under the release', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    const at = centre(shape);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerdown', at.x, at.y));
    });
    expect(document.querySelector('[role="menu"]')).toBeNull();
    await act(() => wait(450));
    await expect.element(menu()).toBeVisible();
    expect(shape.getAttribute('aria-expanded')).toBe('true');

    // Pointer capture routes the moves and the release to the button; the
    // coordinates say which entry they are over.
    const triangle = (await entry('Triangle').element()) as HTMLElement;
    const over = centre(triangle);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointermove', over.x, over.y));
    });
    expect(triangle.hasAttribute('data-highlighted')).toBe(true);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerup', over.x, over.y));
    });
    expect(fake.shapeKind).toBe('triangle');
    expect(fake.activeTool).toBe('shape');
    await expect.element(menu()).not.toBeInTheDocument();
  });

  it('cancels a hold dragged off the slot before the menu opens', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    const at = centre(shape);
    const rect = shape.getBoundingClientRect();
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerdown', at.x, at.y));
      shape.dispatchEvent(pointerAt('pointermove', rect.right + 40, at.y));
    });
    await act(() => wait(450));
    expect(document.querySelector('[role="menu"]')).toBeNull();
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerup', rect.right + 40, at.y));
    });
    expect(fake.activeTool).toBe('view');
  });

  it('keeps the menu open after a hold released on the button, then selects by click', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    const at = centre(shape);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerdown', at.x, at.y));
    });
    await act(() => wait(450));
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerup', at.x, at.y));
    });
    await expect.element(menu()).toBeVisible();
    expect(fake.activeTool).toBe('view');
    await act(() => userEvent.click(entry('Star')));
    expect(fake.shapeKind).toBe('star');
    expect(fake.activeTool).toBe('shape');
    await expect.element(menu()).not.toBeInTheDocument();
  });

  it('lists every subtool as a labelled entry beside the strip and checks the current one', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await openByContextMenu(shape);
    const names = Array.from(document.querySelectorAll('[role="menuitemradio"]')).map((item) =>
      item.textContent?.trim()
    );
    expect(names).toEqual(['Rectangle', 'Ellipse', 'Triangle', 'Star', 'Polygon', 'Freehand']);
    // Opens rightward, clear of the strip, starting level with the slot.
    const content = (await menu().element()) as HTMLElement;
    const strip = (await page.getByRole('toolbar').element()) as HTMLElement;
    await expect
      .poll(() => content.getBoundingClientRect().left)
      .toBeGreaterThanOrEqual(strip.getBoundingClientRect().right);
    await expect
      .poll(() => Math.abs(content.getBoundingClientRect().top - shape.getBoundingClientRect().top))
      .toBeLessThan(8);

    await expect.element(entry('Rectangle')).toHaveAttribute('aria-checked', 'true');
    await act(() => userEvent.click(entry('Ellipse')));
    expect(fake.shapeKind).toBe('ellipse');
    await expect.element(menu()).not.toBeInTheDocument();

    // The slot now stands for the ellipse: reopen and the check moved with it.
    await openByContextMenu(shape);
    await expect.element(entry('Ellipse')).toHaveAttribute('aria-checked', 'true');
    await expect.element(entry('Rectangle')).toHaveAttribute('aria-checked', 'false');
  });

  it('closes on Escape and returns focus to the slot', async () => {
    await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await openByContextMenu(shape);
    await act(() => userEvent.keyboard('{Escape}'));
    await expect.element(menu()).not.toBeInTheDocument();
    await expect.poll(() => document.activeElement).toBe(shape);
  });

  it('closes on an outside press, which still reaches its target', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await openByContextMenu(shape);
    await act(() => userEvent.click(button('Brush')));
    await expect.element(menu()).not.toBeInTheDocument();
    expect(fake.activeTool).toBe('brush');
  });

  it('opens from the keyboard with ArrowRight, navigates, selects, and ArrowLeft returns to the slot', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    // Workbench hotkeys (ArrowRight nudges the selected layer) listen on window.
    const windowKeys: string[] = [];
    const recordKey = (event: KeyboardEvent) => windowKeys.push(event.key);
    window.addEventListener('keydown', recordKey);
    await act(async () => {
      shape.focus();
      await userEvent.keyboard('{ArrowRight}');
    });
    window.removeEventListener('keydown', recordKey);
    expect(windowKeys).not.toContain('ArrowRight');
    await expect.element(menu()).toBeVisible();
    await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('menu');
    await expect.element(entry('Rectangle')).toHaveAttribute('data-highlighted');

    await act(() => userEvent.keyboard('{ArrowDown}'));
    await expect.element(entry('Ellipse')).toHaveAttribute('data-highlighted');
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    await expect.element(menu()).not.toBeInTheDocument();
    await expect.poll(() => document.activeElement).toBe(shape);
    expect(fake.shapeKind).toBe('rect');

    await act(() => userEvent.keyboard('{ArrowRight}'));
    await expect.element(entry('Rectangle')).toHaveAttribute('data-highlighted');
    await act(() => userEvent.keyboard('{ArrowDown}{ArrowDown}'));
    await expect.element(entry('Triangle')).toHaveAttribute('data-highlighted');
    // Zag refocuses the menu a frame after each arrow key; a person cannot press Enter inside that frame.
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );
    await act(() => userEvent.keyboard('{Enter}'));
    expect(fake.shapeKind).toBe('triangle');
    expect(fake.activeTool).toBe('shape');
    await expect.element(menu()).not.toBeInTheDocument();
    await expect.poll(() => document.activeElement).toBe(shape);
  });

  it('remembers the last-used selection tool across tool switches', async () => {
    const fake = await renderStrip();
    await act(() => userEvent.click(button('Marquee select')));
    expect(fake.activeTool).toBe('marquee');

    const select = (await button('Marquee select').element()) as HTMLElement;
    await openByContextMenu(select);
    await act(() => userEvent.click(entry('Lasso select')));
    expect(fake.activeTool).toBe('lasso');

    // Leave the family and come back with a plain click: the slot restores lasso.
    await act(() => userEvent.click(button('Brush')));
    expect(fake.activeTool).toBe('brush');
    await act(() => userEvent.click(button('Lasso select')));
    expect(fake.activeTool).toBe('lasso');
  });

  it('disables the open menu entries when an interaction lock engages', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await openByContextMenu(shape);
    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <ToolStrip isInteractionLocked engine={fake.engine} />
        </ChakraProvider>
      );
    });
    await expect.element(entry('Star')).toHaveAttribute('aria-disabled', 'true');
    await act(() => userEvent.click(entry('Star'), { force: true }));
    expect(fake.shapeKind).toBe('rect');
  });

  it('disables the family slots under an interaction lock, hold included', async () => {
    await renderStrip({ isInteractionLocked: true });
    const shape = (await button('Shape').element()) as HTMLElement;
    expect((shape as HTMLButtonElement).disabled).toBe(true);
    const at = centre(shape);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerdown', at.x, at.y));
    });
    await act(() => wait(450));
    expect(document.querySelector('[role="menu"]')).toBeNull();
    await act(() => {
      shape.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });
});
