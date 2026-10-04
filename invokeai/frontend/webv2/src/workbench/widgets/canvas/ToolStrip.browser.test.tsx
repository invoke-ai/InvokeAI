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

  it('selects the current subtool on a plain click and shows no flyout', async () => {
    const fake = await renderStrip();
    await act(() => userEvent.click(button('Shape')));
    expect(fake.activeTool).toBe('shape');
    expect(fake.shapeKind).toBe('rect');
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('opens on hold and selects the entry under the released pointer', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    const at = centre(shape);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerdown', at.x, at.y));
    });
    expect(document.querySelector('[role="menu"]')).toBeNull();
    await act(() => wait(450));
    expect(document.querySelector('[role="menu"]')).not.toBeNull();

    // Pointer capture routes the release to the button; the coordinates say
    // which entry it landed on.
    const triangle = (await page.getByRole('menuitemradio', { name: 'Triangle' }).element()) as HTMLElement;
    const over = centre(triangle);
    await act(() => {
      shape.dispatchEvent(pointerAt('pointerup', over.x, over.y));
    });
    expect(fake.shapeKind).toBe('triangle');
    expect(fake.activeTool).toBe('shape');
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('keeps the flyout open after a hold released on the button, then selects by click', async () => {
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
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    await act(() => userEvent.click(page.getByRole('menuitemradio', { name: 'Star' })));
    expect(fake.shapeKind).toBe('star');
    expect(fake.activeTool).toBe('shape');
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('marks the slot current entry checked, follows it on the strip icon, and cycles state via right-click', async () => {
    const fake = await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await act(() => {
      shape.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    await expect
      .element(page.getByRole('menuitemradio', { name: 'Rectangle' }))
      .toHaveAttribute('aria-checked', 'true');
    await act(() => userEvent.click(page.getByRole('menuitemradio', { name: 'Ellipse' })));
    expect(fake.shapeKind).toBe('ellipse');

    // The slot now stands for the ellipse: reopen and the check moved with it.
    await act(() => {
      shape.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.element(page.getByRole('menuitemradio', { name: 'Ellipse' })).toHaveAttribute('aria-checked', 'true');
    await expect
      .element(page.getByRole('menuitemradio', { name: 'Rectangle' }))
      .toHaveAttribute('aria-checked', 'false');
  });

  it('closes on Escape and returns focus to the slot', async () => {
    await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await act(() => {
      shape.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    await act(() => userEvent.keyboard('{Escape}'));
    expect(document.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(shape);
  });

  it('opens from the keyboard with ArrowRight and focuses the first entry', async () => {
    await renderStrip();
    const shape = (await button('Shape').element()) as HTMLElement;
    await act(async () => {
      shape.focus();
      await userEvent.keyboard('{ArrowRight}');
      await wait(50);
    });
    expect(document.querySelector('[role="menu"]')).not.toBeNull();
    expect((document.activeElement as HTMLElement | null)?.dataset.subtoolId).toBe('rect');
  });

  it('remembers the last-used selection tool across tool switches', async () => {
    const fake = await renderStrip();
    await act(() => userEvent.click(button('Marquee select')));
    expect(fake.activeTool).toBe('marquee');

    const select = (await button('Marquee select').element()) as HTMLElement;
    await act(() => {
      select.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await act(() => userEvent.click(page.getByRole('menuitemradio', { name: 'Lasso select' })));
    expect(fake.activeTool).toBe('lasso');

    // Leave the family and come back with a plain click: the slot restores lasso.
    await act(() => userEvent.click(button('Brush')));
    expect(fake.activeTool).toBe('brush');
    await act(() => userEvent.click(button('Lasso select')));
    expect(fake.activeTool).toBe('lasso');
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
