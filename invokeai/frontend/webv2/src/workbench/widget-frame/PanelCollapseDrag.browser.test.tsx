/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
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

/**
 * The drag-to-collapse gesture against the *real* store, reducer and shell
 * gating — the unit tests either side of this one mock the command layer, so
 * between them they cannot catch a break in the wiring itself.
 */

const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));

vi.mock('@workbench/WorkbenchContext', () => ({
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

const getRegion = (region: 'bottom' | 'left' | 'right') =>
  storeRef.current!.getSnapshot().activeProject.widgetRegions[region];

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:800px;width:1600px;';
  document.body.append(host);
  root = createRoot(host);
  storeRef.current = createWorkbenchStore();

  // Presets ship the bottom strip closed *and* collapsed, and `setRegionCollapsed`
  // deliberately does not touch `panels.isBottomOpen`. Open it the way the app
  // does — by selecting its active widget from the status rail.
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

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: 260 - startSizePx })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: 260 - startSizePx })));

    expect(getRegion('left').isCollapsed).toBe(true);
    expect(getRegion('left').sizePx).toBe(startSizePx);
    expect(host?.querySelector('[data-testid="collapsed"]')).not.toBeNull();
  });

  it('collapses the right panel, whose axis runs the other way', async () => {
    const separator = await renderShell('right');
    const startSizePx = getRegion('right').sizePx;

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: startSizePx - 260 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: startSizePx - 260 })));

    expect(getRegion('right').isCollapsed).toBe(true);
  });

  it('collapses the bottom strip against its own floor', async () => {
    const separator = await renderShell('bottom');
    const startSizePx = getRegion('bottom').sizePx;

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientY: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientY: startSizePx - 10 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientY: startSizePx - 10 })));

    expect(getRegion('bottom').isCollapsed).toBe(true);
  });

  it('resizes without collapsing when the drag stops at the floor', async () => {
    const separator = await renderShell('left');
    const startSizePx = getRegion('left').sizePx;

    await interact(() => separator.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 0 })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointermove', { clientX: 340 - startSizePx })));
    await interact(() => window.dispatchEvent(new PointerEvent('pointerup', { clientX: 340 - startSizePx })));

    expect(getRegion('left').isCollapsed).toBe(false);
    expect(getRegion('left').sizePx).toBe(350);
  });
});
