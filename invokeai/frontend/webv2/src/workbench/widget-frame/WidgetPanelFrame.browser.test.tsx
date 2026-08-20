/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { FocusRegionProvider } from '@workbench/focusRegions';
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

// The frame reads the region's persisted size and writes back through the
// layout commands; both are stubbed so the drag arithmetic is what is under
// test, not the reducer.
vi.mock('@workbench/WorkbenchContext', () => ({
  shallowEqual: Object.is,
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
          <FocusRegionProvider>
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

/** Drags the handle to a target panel width and releases, unless told not to. */
const dragTo = async (
  separator: Element,
  { end = 'pointerup', widthPx }: { end?: 'pointercancel' | 'none' | 'pointerup'; widthPx: number }
) => {
  await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
  await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: widthPx - frameMocks.sizePx })));

  if (end !== 'none') {
    await interact(() => window.dispatchEvent(new PointerEvent(end, { clientX: widthPx - frameMocks.sizePx })));
  }
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
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WidgetPanelFrame resize', () => {
  it('stops at the floor and commits it', async () => {
    const separator = await renderFrame();

    await dragTo(separator, { widthPx: 300 });

    expect(frameMocks.setRegionSize).toHaveBeenCalledExactlyOnceWith('left', 350);
    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();
  });

  it('collapses instead of resizing once the drag clears the floor by the overshoot', async () => {
    const separator = await renderFrame();

    await dragTo(separator, { widthPx: 260 });

    expect(frameMocks.setRegionCollapsed).toHaveBeenCalledExactlyOnceWith('left', true);
    // The width the user chose survives the collapse, so the rail button
    // reopens the panel where they left it rather than at the floor.
    expect(frameMocks.setRegionSize).not.toHaveBeenCalled();
  });

  it('disarms when the drag comes back inside the floor', async () => {
    const separator = await renderFrame();

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: -250 })));
    expect(separator.hasAttribute('data-collapse-armed')).toBe(true);

    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: -50 })));
    expect(separator.hasAttribute('data-collapse-armed')).toBe(false);

    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: -50 })));
    expect(frameMocks.setRegionSize).toHaveBeenCalledExactlyOnceWith('left', 400);
    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();
  });

  it('never reports a sub-minimum size to assistive tech while armed', async () => {
    const separator = await renderFrame();

    await dragTo(separator, { end: 'none', widthPx: 200 });

    expect(separator.getAttribute('aria-valuenow')).toBe('350');
    expect(separator.getAttribute('aria-valuemin')).toBe('350');
    expect(separator.getAttribute('aria-valuemax')).toBe('720');
  });

  it('drops the window listeners when the frame unmounts mid-drag', async () => {
    const separator = await renderFrame();

    await dragTo(separator, { end: 'none', widthPx: 200 });
    await interact(() => root?.unmount());
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: -250 })));

    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();
    expect(frameMocks.setRegionSize).not.toHaveBeenCalled();
  });

  it('treats a cancelled gesture as an interruption rather than a collapse', async () => {
    const separator = await renderFrame();

    await dragTo(separator, { end: 'pointercancel', widthPx: 200 });

    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();
    expect(frameMocks.setRegionSize).toHaveBeenCalledExactlyOnceWith('left', 350);
  });

  it('mirrors the axis and floor of the region it frames', async () => {
    const rightSeparator = await renderFrame('right');

    // The right panel grows leftwards, so the same pointer delta has to be
    // read with the opposite sign.
    await interact(() => rightSeparator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: 190 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: 190 })));

    expect(frameMocks.setRegionCollapsed).toHaveBeenCalledExactlyOnceWith('right', true);
  });

  it('measures the bottom strip vertically against its own floor', async () => {
    frameMocks.sizePx = 180;

    const separator = await renderFrame('bottom');

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientY: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientY: 100 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientY: 100 })));

    // 180 − 100 = 80, which is 16 below the 96 floor: not yet the 80px overshoot.
    expect(frameMocks.setRegionCollapsed).not.toHaveBeenCalled();
    expect(frameMocks.setRegionSize).toHaveBeenCalledExactlyOnceWith('bottom', 96);
  });
});
