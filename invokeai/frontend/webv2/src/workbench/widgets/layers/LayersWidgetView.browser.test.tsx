/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { shallowEqual, useExternalStoreSelector } from '@platform/state/selectors';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { resolveWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

type Store = ReturnType<typeof createWorkbenchStore>;
type Snapshot = ReturnType<Store['getSnapshot']>;

const harness = vi.hoisted(() => ({
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const store = () => harness.store!;
const useSnapshot = <Selected,>(
  select: (snapshot: Snapshot) => Selected,
  isEqual: (a: Selected, b: Selected) => boolean = shallowEqual
) => useExternalStoreSelector(store().subscribe, store().getSnapshot, select, isEqual);

// A real store and engine without account persistence or the canvas registry; the panel reads and acts through them.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => useSnapshot((snapshot) => snapshot.activeProject.id),
  useActiveProjectName: () => useSnapshot((snapshot) => snapshot.activeProject.name),
  useActiveProjectSelector: <T,>(select: (project: Project) => T, isEqual?: (a: T, b: T) => boolean) =>
    useSnapshot((snapshot) => select(snapshot.activeProject), isEqual),
  useOptionalWorkbenchCommands: () => store().commands,
  useOptionalWorkbenchQueries: () => store().queries,
  useWidgetInstanceValuesSelector: <T,>(
    instanceId: string,
    select: (values: Record<string, unknown>) => T,
    isEqual?: (a: T, b: T) => boolean
  ) =>
    useSnapshot(
      (snapshot) =>
        select(resolveWidgetValues(snapshot.activeProject.widgetInstances[instanceId], snapshot.scopedDefaults)),
      isEqual
    ),
  useWorkbenchCommands: () => store().commands,
  useWorkbenchQueries: () => store().queries,
  useWorkbenchSelector: <T,>(select: (snapshot: Snapshot) => T, isEqual?: (a: T, b: T) => boolean) =>
    useSnapshot(select, isEqual),
}));
vi.mock('@workbench/canvas-operations/react', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useCanvasEngine: () => harness.engine,
}));

const { LayersWidgetView } = await import('./LayersWidgetView');
const { PanelBodySlot } = await import('@workbench/widget-frame/WidgetRenderer');
const { layersWidgetManifest } = await import('./manifest');

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  initAsync: false,
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const noop = () => undefined;
const runtime = {
  commands: { register: () => noop },
  hotkeys: { register: () => noop },
  instanceId: 'layers-instance',
  state: { patch: noop },
} as unknown as WidgetViewProps['runtime'];
const manifest = { id: 'layers', label: 'Layers' } as unknown as WidgetViewProps['manifest'];
const instance = { id: 'layers-instance', typeId: 'layers' } as unknown as WidgetViewProps['instance'];

/** The Add-layer menu the panel opens for a right-click on its empty area. */
const addLayerMenus = () =>
  [...document.querySelectorAll<HTMLElement>('[role="menu"][data-state="open"]')].filter(
    (menu) => menu.querySelector('[role="menuitem"][data-value="inpaint_mask"]') !== null
  );

let host: HTMLDivElement;
let root: Root;

/** Mounts the panel over a real store and engine; `wrap` places it in a host such as the widget frame's body. */
const mountPanel = async ({
  hostStyle = 'display:flex;flex-direction:column;height:700px;width:360px',
  layers = [layerContract('art', 'raster', { name: 'Art' })],
  wrap = (view) => view,
}: {
  hostStyle?: string;
  layers?: CanvasNodeContract[];
  wrap?: (view: ReactNode) => ReactNode;
} = {}) => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
  const document = { ...base.canvas.document, stacks: stacksFrom(layers) };
  harness.store = createWorkbenchStore({ ...initial, projects: [{ ...base, canvas: { ...base.canvas, document } }] });
  const projectId = store().getState().activeProjectId;
  harness.engine = createCanvasEngine({
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.reject(new Error('no images in this harness')),
    mutationPort: createCanvasProjectMutationPort(store(), projectId),
    projectId,
    reportError: noop,
  });
  host = window.document.createElement('div');
  host.style.cssText = hostStyle;
  window.document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <QueryClientProvider client={new QueryClient()}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            {wrap(<LayersWidgetView instance={instance} manifest={manifest} region="right" runtime={runtime} />)}
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    )
  );
  await expect.element(page.getByRole('treeitem', { name: /Art/ })).toBeVisible();
};

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  harness.engine?.lifecycle.dispose();
  harness.engine = null;
  harness.store = null;
});

/** Menus open within a frame of their event; give a leaked one two frames to show before checking. */
const expectNoAddLayerMenu = async () => {
  for (let frame = 0; frame < 2; frame++) {
    await new Promise((resolve) => {
      requestAnimationFrame(resolve);
    });
  }
  expect(addLayerMenus()).toHaveLength(0);
};

describe('layers panel context menus', () => {
  beforeEach(() => mountPanel());

  it("leaves a right-click in a layer menu's rename dialog to the browser", async () => {
    await userEvent.click(page.getByRole('treeitem', { name: /Art/ }), { button: 'right' });
    await page.getByRole('menuitem', { name: 'Rename layer', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Rename layer' });
    await expect.element(dialog).toBeVisible();

    const seen: { prevented: boolean | null } = { prevented: null };
    const record = (event: Event) => {
      seen.prevented = event.defaultPrevented;
    };
    window.addEventListener('contextmenu', record);
    try {
      await userEvent.click(dialog.getByRole('textbox'), { button: 'right' });
    } finally {
      window.removeEventListener('contextmenu', record);
    }

    // The input's native menu (paste, spelling) applies: nothing on the page claimed the event.
    expect(seen.prevented).toBe(false);
    await expectNoAddLayerMenu();
    await expect.element(dialog).toBeVisible();
  });

  it("keeps a right-click on an open layer menu's item inside that menu", async () => {
    await userEvent.click(page.getByRole('treeitem', { name: /Art/ }), { button: 'right' });
    const rename = page.getByRole('menuitem', { name: 'Rename layer', exact: true });
    await expect.element(rename).toBeVisible();

    await userEvent.click(rename, { button: 'right' });

    await expectNoAddLayerMenu();
  });

  it('opens the Add-layer menu for a right-click on the empty area below the layers', async () => {
    const tree = page.getByRole('tree').element();
    // The tree's scroll area keeps a minimum height, so space below the last row stays part of the panel.
    const area = tree.parentElement!.parentElement!;
    const bounds = area.getBoundingClientRect();
    const lastRow = [...tree.querySelectorAll('[role="treeitem"]')].at(-1)!.getBoundingClientRect();
    expect(bounds.bottom - lastRow.bottom).toBeGreaterThan(20);

    await userEvent.click(area, {
      button: 'right',
      position: { x: bounds.width / 2, y: (lastRow.bottom + bounds.bottom) / 2 - bounds.top },
    });

    await expect.poll(() => addLayerMenus()).toHaveLength(1);
  });
});

describe('layers panel in a docked widget body', () => {
  // Enough rows that the tree's own height is several times the panel's.
  const manyLayers = Array.from({ length: 65 }, (_, index) =>
    layerContract(`layer-${index}`, 'raster', { name: index === 0 ? 'Art' : `Layer ${index}` })
  );
  const mountInPanelBody = (heightPx: number) =>
    mountPanel({
      hostStyle: `display:flex;flex-direction:column;height:${heightPx}px;width:360px`,
      layers: manyLayers,
      wrap: (view) => <PanelBodySlot body={layersWidgetManifest.chrome?.body}>{view}</PanelBodySlot>,
    });
  const bodyViewport = () => host.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
  const treeViewport = () => page.getByRole('tree').element().closest<HTMLElement>('[data-part="viewport"]')!;
  const editorPane = () => host.querySelector<HTMLElement>('[data-layer-pane-block="layer-editor-pane"]')!;

  it('fits the panel to the body, so the editor pane stays in view and the tree scrolls itself', async () => {
    await mountInPanelBody(700);
    const viewport = bodyViewport();
    const bounds = viewport.getBoundingClientRect();
    const panelRoot = editorPane().parentElement!;

    expect(panelRoot.getBoundingClientRect().height).toBeCloseTo(bounds.height, 0);
    // The Properties / Transform / Overview block sits wholly inside the body, its panel open.
    const pane = editorPane().getBoundingClientRect();
    expect(pane.top).toBeGreaterThanOrEqual(bounds.top);
    expect(pane.bottom).toBeLessThanOrEqual(bounds.bottom + 0.5);
    expect(editorPane().querySelector('[role="tabpanel"]')!.getBoundingClientRect().height).toBeGreaterThan(40);
    // The rows scroll inside the tree, and the body has nothing of its own to scroll.
    expect(treeViewport().scrollHeight).toBeGreaterThan(treeViewport().clientHeight);
    expect(viewport.scrollHeight).toBeLessThanOrEqual(viewport.clientHeight);
  });

  it("scrolls a dock shorter than the panel's minimums instead of crushing the tree", async () => {
    await mountInPanelBody(260);
    const viewport = bodyViewport();
    const treePanel = page.getByRole('tree').element().closest<HTMLElement>('[role="tabpanel"]')!;

    // The tree keeps its 13rem floor; the body scrolls to reach the panes below it.
    expect(treePanel.getBoundingClientRect().height).toBeGreaterThanOrEqual(208);
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    viewport.scrollTop = viewport.scrollHeight;
    expect(editorPane().getBoundingClientRect().bottom).toBeLessThanOrEqual(
      viewport.getBoundingClientRect().bottom + 0.5
    );
  });

  it('never squeezes the open colour pane below its whole wheel, however short the dock', async () => {
    await mountInPanelBody(260);
    const colorPane = host.querySelector<HTMLElement>('[data-layer-pane-block="layer-color-pane"]')!;
    const colorPanel = colorPane.querySelector<HTMLElement>('[role="tabpanel"]')!.getBoundingClientRect();
    // The hue ring spans the wheel's full box.
    const wheel = colorPane.querySelector<HTMLElement>('[role="slider"]')!.getBoundingClientRect();

    expect(wheel.height).toBe(168);
    expect(wheel.top).toBeGreaterThanOrEqual(colorPanel.top);
    expect(wheel.bottom).toBeLessThanOrEqual(colorPanel.bottom);
  });

  it('folds the colour pane to its tabs while a drag is past the collapse point, before the button lifts', async () => {
    await mountInPanelBody(700);
    const colorPane = host.querySelector<HTMLElement>('[data-layer-pane-block="layer-color-pane"]')!;
    // The colour pane docks at the top, so its divider is the panel's first.
    const separator = host.querySelector<HTMLElement>('[role="separator"]')!;
    const pointer = (type: string, clientY: number, buttons = 1) =>
      new PointerEvent(type, { bubbles: true, buttons, clientY, pointerId: 1 });
    const frame = () =>
      new Promise((resolve) => {
        requestAnimationFrame(resolve);
      });

    const expandedHeight = colorPane.getBoundingClientRect().height;
    await act(() => separator.dispatchEvent(pointer('pointerdown', 300)));

    // Short of the collapse point the pane holds its floor, the wheel whole.
    await act(() => window.dispatchEvent(pointer('pointermove', 280)));
    await frame();
    expect(colorPane.getBoundingClientRect().height).toBeGreaterThanOrEqual(224);

    // Past it, the pane shows only its tab strip while the button is still down.
    await act(() => window.dispatchEvent(pointer('pointermove', 150)));
    await frame();
    expect(expandedHeight).toBeGreaterThanOrEqual(224);
    expect(colorPane.getBoundingClientRect().height).toBe(40);
    expect(colorPane.hasAttribute('data-pane-collapsed')).toBe(false);

    // Releasing commits the collapse through the divider (ResizeHandle's own tests); here it only ends the drag.
    await act(() => window.dispatchEvent(pointer('pointerup', 150, 0)));
  });
});
