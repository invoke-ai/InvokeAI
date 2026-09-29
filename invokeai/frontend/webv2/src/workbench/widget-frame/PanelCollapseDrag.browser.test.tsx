/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type * as workbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/** Exercise real store/reducer/shell wiring absent from command-mocked unit tests. */

const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  shallowEqual: Object.is,
  useActiveProjectSelector: (selector: (project: never) => unknown) => {
    const store = storeRef.current!;
    const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

    return selector(snapshot.activeProject as never);
  },
  useWorkbenchCommands: () => storeRef.current!.commands,
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

/** Mirrors how `WorkbenchShell`/`BottomPanel` decide to mount a panel at all. */
const Shell = ({ region }: { region: 'bottom' | 'left' | 'right' }) => {
  const store = storeRef.current!;
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  const regionState = snapshot.activeProject.widgetRegions[region];
  const panels = snapshot.activeProject.layout.panels;
  const isOpen = region === 'left' ? panels.isLeftOpen : region === 'right' ? panels.isRightOpen : panels.isBottomOpen;

  if (!isOpen || regionState.isCollapsed) {
    return <div data-testid="collapsed" />;
  }

  return (
    <WidgetPanelFrame instanceId={regionState.activeInstanceId} region={region} typeId="gallery">
      <div />
    </WidgetPanelFrame>
  );
};

const renderShell = async (region: 'bottom' | 'left' | 'right') => {
  await interact(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider>
            <Shell region={region} />
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

/** A real gesture: buttons held while moving, one frame for the move to land, then release. */
const drag = async (separator: Element, axis: 'clientX' | 'clientY', delta: number) => {
  const pointer = (type: string, at: number, buttons: number) =>
    new PointerEvent(type, { bubbles: true, buttons, pointerId: 1, [axis]: at });

  await interact(() => separator.dispatchEvent(pointer('pointerdown', 0, 1)));
  await interact(() => window.dispatchEvent(pointer('pointermove', delta, 1)));
  await act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      })
  );
  await interact(() => window.dispatchEvent(pointer('pointerup', delta, 0)));
};

const getRegion = (region: 'bottom' | 'left' | 'right') =>
  storeRef.current!.getSnapshot().activeProject.widgetRegions[region];

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:800px;width:1600px;';
  document.body.append(host);
  root = createRoot(host);
  storeRef.current = createWorkbenchStore();

  // Open the bottom strip through its active status widget; setRegionCollapsed does not set isBottomOpen.
  const bottom = storeRef.current.getSnapshot().activeProject.widgetRegions.bottom;

  storeRef.current.commands.widgets.select({
    projectId: storeRef.current.getSnapshot().activeProject.id,
    region: 'bottom',
    widgetId: bottom.activeInstanceId,
  });
});

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  storeRef.current = null;
});

describe('drag-to-collapse against the real aggregate', () => {
  it('collapses the left panel and unmounts it', async () => {
    const separator = await renderShell('left');
    const startSizePx = getRegion('left').sizePx;

    await drag(separator, 'clientX', 260 - startSizePx);

    expect(getRegion('left').isCollapsed).toBe(true);
    expect(getRegion('left').sizePx).toBe(startSizePx);
    expect(host?.querySelector('[data-testid="collapsed"]')).not.toBeNull();
  });

  it('collapses the right panel, whose axis runs the other way', async () => {
    const separator = await renderShell('right');
    const startSizePx = getRegion('right').sizePx;

    await drag(separator, 'clientX', startSizePx - 260);

    expect(getRegion('right').isCollapsed).toBe(true);
  });

  it('collapses the bottom strip against its own floor', async () => {
    const separator = await renderShell('bottom');
    const startSizePx = getRegion('bottom').sizePx;

    await drag(separator, 'clientY', startSizePx - 10);

    expect(getRegion('bottom').isCollapsed).toBe(true);
  });

  it('resizes without collapsing when the drag stops at the floor', async () => {
    const separator = await renderShell('left');
    const startSizePx = getRegion('left').sizePx;

    await drag(separator, 'clientX', 340 - startSizePx);

    expect(getRegion('left').isCollapsed).toBe(false);
    expect(getRegion('left').sizePx).toBe(350);
  });
});
