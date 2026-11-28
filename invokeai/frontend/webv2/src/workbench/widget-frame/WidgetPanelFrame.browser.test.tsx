/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const frameMocks = vi.hoisted(() => ({
  setRegionCollapsed: vi.fn(),
  setRegionSize: vi.fn(),
  sizePx: 450,
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  shallowEqual: Object.is,
  useActiveLayoutSelector: (
    selector: (project: {
      widgetRegions: Record<string, { activeInstanceId: string; instanceIds: string[]; sizePx: number }>;
    }) => unknown
  ) => {
    const region = {
      activeInstanceId: 'test-instance',
      instanceIds: ['test-instance'],
      sizePx: frameMocks.sizePx,
    };

    return selector({ widgetRegions: { bottom: region, center: region, left: region, right: region } });
  },
  useActiveProjectSelector: (
    selector: (project: {
      widgetRegions: Record<string, { activeInstanceId: string; instanceIds: string[]; sizePx: number }>;
    }) => unknown
  ) => {
    const region = {
      activeInstanceId: 'test-instance',
      instanceIds: ['test-instance'],
      sizePx: frameMocks.sizePx,
    };

    return selector({ widgetRegions: { bottom: region, center: region, left: region, right: region } });
  },
  useWorkbenchCommands: () => ({
    layout: { setRegionCollapsed: frameMocks.setRegionCollapsed, setRegionSize: frameMocks.setRegionSize },
  }),
}));

import { WidgetPanelFrame } from './WidgetFrames';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: { widgets: { panelLabel: '{{region}} panel' } } } },
});

const focusController = createTestFocusController();

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (run: () => void): Promise<void> =>
  act(async () => {
    run();
    await Promise.resolve();
  });

const renderFrame = async (region: 'bottom' | 'left' | 'right' = 'left') => {
  await interact(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={focusController}>
            <WidgetPanelFrame instanceId="test-instance" region={region} typeId="gallery">
              <div />
            </WidgetPanelFrame>
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

  const separator = host?.querySelector('[role="separator"]');

  if (!separator) {
    throw new Error('panel frame did not render a resize handle');
  }

  return separator;
};

const pointer = (type: string, init: PointerEventInit, buttons = 1) =>
  new PointerEvent(type, { bubbles: true, buttons, pointerId: 1, ...init });

const nextFrame = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );

const holdDrag = async (separator: Element, delta: number) => {
  await interact(() => separator.dispatchEvent(pointer('pointerdown', { clientX: 0 })));
  await interact(() => window.dispatchEvent(pointer('pointermove', { clientX: delta })));
  await nextFrame();
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:600px;width:1400px;';
  document.body.append(host);
  root = createRoot(host);
  frameMocks.setRegionCollapsed.mockClear();
  frameMocks.setRegionSize.mockClear();
  frameMocks.sizePx = 450;
});

afterEach(async () => {
  focusController.clear();
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WidgetPanelFrame resize', () => {
  it('previews the collapse by closing the panel on screen', async () => {
    const separator = await renderFrame();
    const frame = separator.parentElement!.parentElement!;

    await holdDrag(separator, -200);

    expect(frame.getBoundingClientRect().width).toBe(0);
  });

  it('never reports a sub-minimum size to assistive tech while a collapse is armed', async () => {
    const separator = await renderFrame();

    await holdDrag(separator, -200);
    await nextFrame();

    expect(separator.getAttribute('aria-valuenow')).toBe('450');
    expect(separator.getAttribute('aria-valuemin')).toBe('350');
  });

  it("hides its divider line while the panel's outline is drawn over it", async () => {
    const separator = await renderFrame();
    const divider = separator.parentElement!;

    expect(divider.hasAttribute('data-line-hidden')).toBe(true);
    await interact(() => focusController.activate({ kind: 'region', region: 'right' }));
    expect(divider.hasAttribute('data-line-hidden')).toBe(false);
    await interact(() =>
      host!.querySelector('aside')!.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }))
    );

    expect(divider.hasAttribute('data-line-hidden')).toBe(true);
  });

  it('leaves the edge of the left panel, where its scrollbar sits, to the panel', async () => {
    frameMocks.sizePx = 350;
    const separator = await renderFrame();
    const aside = separator.parentElement!.previousElementSibling!.getBoundingClientRect();
    const probeY = aside.top + aside.height / 4;

    expect(document.elementFromPoint(aside.right - 2, probeY)).not.toBe(separator);
    expect(document.elementFromPoint(aside.right + 5, probeY)).toBe(separator);
  });

  it('collapses on a further collapse-ward key press at the floor', async () => {
    frameMocks.sizePx = 350;
    const separator = await renderFrame();

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));
    expect(frameMocks.setRegionSize).toHaveBeenCalledExactlyOnceWith('left', 366);
    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' })));
    expect(frameMocks.setRegionCollapsed).toHaveBeenCalledExactlyOnceWith('left', true);
  });
});

describe('WidgetPanelFrame squeezed by the viewport', () => {
  const renderSqueezed = async () => {
    host!.style.cssText = 'display:flex;height:600px;width:400px;';
    await interact(() =>
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <FocusRegionProvider controller={focusController}>
              <WidgetPanelFrame instanceId="test-instance" region="left" typeId="gallery">
                <div />
              </WidgetPanelFrame>
              <div style={{ flex: 1, minWidth: '200px' }} />
            </FocusRegionProvider>
          </ChakraProvider>
        </I18nextProvider>
      )
    );
    // The frame learns its on-screen width from a ResizeObserver, which reports after layout.
    await interact(() => {});
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => resolve());
        })
    );

    const separator = host?.querySelector('[role="separator"]');

    if (!separator) {
      throw new Error('panel frame did not render a resize handle');
    }

    return separator;
  };

  it('yields to a rigid sibling instead of pushing it out of the row', async () => {
    const separator = await renderSqueezed();

    expect(separator.parentElement!.parentElement!.getBoundingClientRect().width).toBe(200);
    expect(separator.getAttribute('aria-valuenow')).toBe('200');
    expect(separator.getAttribute('aria-valuemin')).toBe('200');
  });

  it('snaps shut after the overshoot from the width on screen, not from the floor', async () => {
    const separator = await renderSqueezed();

    await holdDrag(separator, -79);
    expect(separator.hasAttribute('data-collapse-armed')).toBe(false);

    await interact(() => window.dispatchEvent(pointer('pointermove', { clientX: -80 })));
    await nextFrame();
    expect(separator.hasAttribute('data-collapse-armed')).toBe(true);

    await interact(() => window.dispatchEvent(pointer('pointerup', { clientX: -80 }, 0)));
    expect(frameMocks.setRegionCollapsed).toHaveBeenCalledExactlyOnceWith('left', true);
    expect(frameMocks.setRegionSize).not.toHaveBeenCalled();
  });

  it('collapses on a collapse-ward key press, being already below the floor', async () => {
    const separator = await renderSqueezed();

    await interact(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowLeft' })));

    expect(frameMocks.setRegionCollapsed).toHaveBeenCalledExactlyOnceWith('left', true);
  });
});
