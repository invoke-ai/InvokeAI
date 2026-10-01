/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type * as fonts from '@features/fonts';
import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/api';
import type { CanvasOperationState } from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { FilterOperationSessionState } from '@workbench/canvas-operations/filterOperationSession';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project } from '@workbench/projectContracts';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import {
  createEngineRegistry,
  type EngineDeps,
  type EngineRegistry,
} from '@workbench/canvas-operations/engineRegistry';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument, createEmptyCanvasState } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { resetPropertyGroupCollapse } from '@workbench/widgets/canvas/tool-presentation/propertyGroupStore';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({
  engine: null as unknown,
  listeners: new Set<() => void>(),
  project: null as Project | null,
}));

// The text form's catalog query must not reach the network from this harness.
vi.mock('@features/fonts', async (importOriginal) => ({
  ...(await importOriginal<typeof fonts>()),
  fontsInfiniteQueryOptions: () => ({
    getNextPageParam: () => undefined,
    initialPageParam: 0,
    queryFn: () => Promise.resolve({ items: [], limit: 100, offset: 0, total: 0 }),
    queryKey: ['font-catalog-pane-test'],
  }),
}));
vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  const subscribe = (listener: () => void) => {
    harness.listeners.add(listener);
    return () => harness.listeners.delete(listener);
  };
  // Merges into the harness project's widget values and notifies, so pair
  // edits re-render the way the real store does.
  const patchValues = (typeId: string, values: Record<string, unknown>) => {
    const project = harness.project!;
    const entry = Object.entries(project.widgetInstances).find(([, instance]) => instance.typeId === typeId);
    if (!entry) {
      return;
    }
    const [instanceId, instance] = entry;
    harness.project = {
      ...project,
      widgetInstances: {
        ...project.widgetInstances,
        [instanceId]: { ...instance, state: { ...instance.state, values: { ...instance.state.values, ...values } } },
      },
    };
    harness.listeners.forEach((listener) => listener());
  };
  const commands = { widgets: { patchValues } };
  const queries = { getSnapshot: () => ({ activeProject: harness.project! }) };
  return {
    useActiveProjectId: () => harness.project!.id,
    useActiveProjectSelector: (selector: (project: Project) => unknown) => {
      const project = useSyncExternalStore(subscribe, () => harness.project!);
      return selector(project);
    },
    useOptionalWorkbenchCommands: () => null,
    useWorkbenchCommands: () => commands,
    useWorkbenchQueries: () => queries,
  };
});
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => () => true,
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));

import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { clearMaskTintTarget } from '@workbench/widgets/canvas/color-system/maskTintTarget';

import type { LayerEditorPaneLayout } from './editorPaneLayout';

import { ColorPane } from './ColorPane';
import { LAYER_EDITOR_PANE_DEFAULTS } from './editorPaneLayout';
import { HistoryPane } from './HistoryPane';
import { LayerEditorPanes } from './LayerEditorPanes';
import { OverviewPane } from './OverviewPane';
import { PropertiesPane } from './PropertiesPane';
import { SwatchesPane } from './SwatchesPane';
import { TransformPane } from './TransformPane';

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The engine reads the harness project's canvas, so document edits resolve against the layers the views show. */
const createEngineDeps = (): EngineDeps => {
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: () => false,
    getCanvasState: () => harness.project?.canvas ?? createEmptyCanvasState(64, 64),
    subscribe: () => () => undefined,
  };
  return {
    getMainModelBase: () => null,
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort,
    reportError: () => undefined,
  };
};

const filterSession = (): FilterOperationSessionState => ({
  autoProcess: true,
  draft: { settings: { high_threshold: 200, low_threshold: 100 }, type: 'canny_edge_detection' },
  error: null,
  initialFilter: null,
  layerId: 'layer-1',
  layerName: 'Portrait',
  layerType: 'raster',
  preview: null,
  status: 'ready',
});

const createFakeOperations = () => {
  const listeners = new Set<() => void>();
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };
  let filter: FilterOperationSessionState | null = null;
  let state: CanvasOperationState = { status: 'idle' };
  return {
    cancelFilterOperation: vi.fn(),
    commitFilterOperation: vi.fn(),
    getFilterSessionState: () => filter,
    getOperationState: () => state,
    getSamSessionState: () => null,
    processFilterOperation: vi.fn(),
    resetFilterOperation: vi.fn(),
    setFilterOperationAutoProcess: vi.fn(),
    start: (running: boolean) => {
      filter = running ? filterSession() : null;
      state = running
        ? {
            error: null,
            identity: { kind: 'filter', layerId: 'layer-1', projectId: 'p' },
            phase: 'ready',
            status: 'active',
          }
        : { status: 'idle' };
      listeners.forEach((listener) => listener());
    },
    subscribeFilterSession: subscribe,
    subscribeOperation: subscribe,
    subscribeSamSession: subscribe,
    updateFilterOperation: vi.fn(),
  };
};

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
let host: HTMLDivElement | null = null;
let root: Root | null = null;
let registry: EngineRegistry | null = null;
let engine: CanvasEngine | null = null;
let operations: ReturnType<typeof createFakeOperations> | null = null;

const settle = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      })
  );

const IDENTITY = { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 };

type Selection = 'none' | 'layer' | 'group' | 'mask' | 'shape' | 'gradient' | 'text';

const SHAPE_SOURCE = {
  fill: '#ff0000',
  height: 10,
  kind: 'rect',
  stroke: null,
  strokeWidth: 2,
  type: 'shape',
  width: 10,
} satisfies Extract<CanvasLayerSourceContract, { type: 'shape' }>;
const TEXT_SOURCE = {
  align: 'center',
  color: '#123456',
  content: 'Hello there\nsecond line',
  fontFamily: 'system-ui, sans-serif',
  fontSize: 72,
  fontWeight: 700,
  lineHeight: 1.5,
  type: 'text',
} satisfies Extract<CanvasLayerSourceContract, { type: 'text' }>;
const GRADIENT_SOURCE = {
  angle: 0,
  height: 10,
  kind: 'linear',
  stops: [
    { color: '#000000ff', offset: 0 },
    { color: '#ffffffff', offset: 1 },
  ],
  type: 'gradient',
  width: 10,
} satisfies Extract<CanvasLayerSourceContract, { type: 'gradient' }>;

const mount = async (View: typeof PropertiesPane, selection: Selection = 'none') => {
  const base = { ...createInitialWorkbenchState().projects[0]!, id: 'p' };
  const nodes =
    selection === 'none'
      ? []
      : selection === 'layer'
        ? [layerContract('l0', 'raster', { name: 'Paint', transform: { ...IDENTITY, rotation: 0.5, x: 10.4 } })]
        : selection === 'shape'
          ? [layerContract('l0', 'raster', { name: 'My Shape', source: SHAPE_SOURCE })]
          : selection === 'gradient'
            ? [layerContract('l0', 'raster', { name: 'My Gradient', source: GRADIENT_SOURCE })]
            : selection === 'text'
              ? [layerContract('l0', 'raster', { name: 'My Text', source: TEXT_SOURCE })]
              : selection === 'mask'
                ? [layerContract('m0', 'inpaint_mask', { name: 'Mask' })]
                : [groupContract('g0', [layerContract('l0', 'raster', { name: 'Paint' })], { name: 'Folder' })];
  harness.project = applyCanvasProjectMutation(base, {
    document: {
      ...createEmptyCanvasDocument(),
      selectedLayerId: selection === 'none' ? null : selection === 'mask' ? 'm0' : selection === 'group' ? 'g0' : 'l0',
      stacks: stacksFrom(nodes),
    },
    type: 'replaceCanvasDocument',
  });
  host = document.createElement('div');
  host.style.cssText = 'width:450px;height:600px;';
  document.body.append(host);
  root = createRoot(host);
  registry = createEngineRegistry({ gracePeriodMs: 0 });
  engine = registry.getOrCreateEngine('p', createEngineDeps());
  operations = createFakeOperations();
  attachCanvasOperations(engine, operations as never);
  harness.engine = engine;
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <View />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  await settle();
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  registry?.releaseEngine('p');
  host = null;
  root = null;
  registry = null;
  engine = null;
  harness.engine = null;
});

let paneHarnessLayout: LayerEditorPaneLayout = { ...LAYER_EDITOR_PANE_DEFAULTS };
const LayerEditorPanesHarness = () => {
  const [layout, setLayout] = useState(paneHarnessLayout);
  return <LayerEditorPanes layout={layout} onLayoutChange={setLayout} />;
};

describe('Properties pane', () => {
  // Collapse overrides persist in localStorage, which the browser session
  // shares across test FILES; reset around every test so no order dependence
  // leaks in either direction.
  beforeEach(() => resetPropertyGroupCollapse());
  afterEach(() => resetPropertyGroupCollapse());

  it('shows the active tool as a grouped form and swaps it with the tool', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    expect(host!.textContent).toContain('Brush');
    await expect.element(page.getByRole('slider', { exact: true, name: 'Brush size' })).toBeVisible();
    await expect.element(page.getByRole('slider', { exact: true, name: 'Opacity' })).toBeVisible();
    await expect.element(page.getByRole('button', { exact: true, name: 'Brush color' })).toBeVisible();
    // Dynamics ships collapsed; the disclosure opens it to labelled switches.
    const dynamics = page.getByRole('button', { exact: true, name: 'Dynamics' });
    await expect.element(dynamics).toHaveAttribute('aria-expanded', 'false');
    expect(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects width' }).query()).toBeNull();
    await act(() => userEvent.click(dynamics));
    await expect.element(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects width' })).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects opacity' }))
      .toBeVisible();

    await act(() => engine!.tools.setTool('view'));
    await settle();
    expect(page.getByRole('slider', { exact: true, name: 'Brush size' }).query()).toBeNull();
    // Hint-only tools show a gesture card now, not a single sentence.
    expect(host!.textContent).toContain('Pan the canvas.');
    expect(host!.textContent).toContain('Hold Space');
  });

  it('keeps a slider and its number field on one grid row and shares row identity brush↔eraser', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    const slider = page.getByRole('slider', { exact: true, name: 'Brush size' }).element() as HTMLElement;
    const field = page.getByRole('spinbutton', { exact: true, name: 'Brush size' }).element() as HTMLElement;
    // One grid row: the slider and its field never wrap apart (compare row
    // centers; the two controls have different heights).
    const centerOf = (el: HTMLElement) => {
      const rect = el.getBoundingClientRect();
      return rect.top + rect.height / 2;
    };
    expect(Math.abs(centerOf(slider) - centerOf(field))).toBeLessThan(8);

    await act(() => engine!.tools.setTool('eraser'));
    await settle();
    // Same DOM node, relabelled: the shared Stroke group survives the switch.
    const eraserSlider = page.getByRole('slider', { exact: true, name: 'Eraser size' }).element();
    expect(eraserSlider).toBe(slider);
    expect(page.getByRole('button', { exact: true, name: 'Brush color' }).query()).toBeNull();
  });

  it('labels the shape form with its edit target and gates stroke width on the stroke slot', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('shape'));
    await settle();
    // Nothing selected: the chip says the form edits the creation defaults.
    expect(host!.textContent).toContain('Defaults');
    await expect.element(page.getByRole('radio', { exact: true, name: 'Rectangle' })).toBeInTheDocument();
    await expect.element(page.getByRole('radio', { exact: true, name: 'Triangle' })).toBeInTheDocument();
    await expect.element(page.getByRole('radio', { exact: true, name: 'Star' })).toBeInTheDocument();

    await act(() => root?.unmount());
    await mount(PropertiesPane, 'shape');
    await act(() => engine!.tools.setTool('shape'));
    await settle();
    expect(host!.textContent).toContain('Editing: My Shape');
    // The fixture shape has no stroke, so the width slider is disabled in place.
    const width = page.getByRole('slider', { exact: true, name: 'Stroke width' });
    await expect.element(width).toBeVisible();
    expect(width.element().getAttribute('data-disabled')).not.toBeNull();
  });

  it('previews the selected text layer above its form and offers style and weight on one row', async () => {
    await mount(PropertiesPane, 'text');
    await act(() => engine!.tools.setTool('text'));
    await settle();
    await expect.element(page.getByRole('combobox', { exact: true, name: 'Style' })).toBeVisible();
    // The specimen card carries the chip and the layer's first line in its own style.
    expect(host!.textContent).not.toContain('second line');
    const specimen = Array.from(host!.querySelectorAll<HTMLElement>('[aria-hidden="true"]')).find((el) =>
      el.textContent?.startsWith('Hello there')
    )!;
    expect(specimen.parentElement!.textContent).toContain('Editing: My Text');
    expect(getComputedStyle(specimen).fontWeight).toBe('700');
    expect(getComputedStyle(specimen).textAlign).toBe('center');
    expect(getComputedStyle(specimen).color).toBe('rgb(18, 52, 86)');
    const style = page.getByRole('combobox', { exact: true, name: 'Style' }).element();
    const weight = page.getByRole('combobox', { exact: true, name: 'Weight' }).element();
    expect(Math.abs(style.getBoundingClientRect().top - weight.getBoundingClientRect().top)).toBeLessThan(2);
    expect((page.getByRole('spinbutton', { exact: true, name: 'Font size' }).element() as HTMLInputElement).value).toBe(
      '72'
    );
  });

  it('steps the log-scaled text size slider by whole pixels from the keyboard', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('text'));
    await settle();
    const slider = page.getByRole('slider', { exact: true, name: 'Font size' });
    await expect.element(slider).toBeVisible();
    const field = () =>
      (page.getByRole('spinbutton', { exact: true, name: 'Font size' }).element() as HTMLInputElement).value;
    expect(field()).toBe('48');
    await act(() => (slider.element() as HTMLElement).focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(field()).toBe('49');
    await act(() => userEvent.keyboard('{PageUp}'));
    expect(field()).toBe('59');
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(field()).toBe('58');
    expect(engine!.interaction.get('textOptions').fontSize).toBe(58);
  });

  it('names a selected gradient in the chip and moves a default stop from the keyboard', async () => {
    await mount(PropertiesPane, 'gradient');
    await act(() => engine!.tools.setTool('gradient'));
    await settle();
    expect(host!.textContent).toContain('Editing: My Gradient');
    await expect.element(page.getByRole('button', { exact: true, name: 'Gradient stop at 0%' })).toBeVisible();

    // The harness's mutation port refuses document dispatches, so the
    // keyboard-move half runs against the creation DEFAULTS (options store).
    await act(() => root?.unmount());
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('gradient'));
    await settle();
    expect(host!.textContent).toContain('Defaults');
    const startStop = page.getByRole('button', { exact: true, name: 'Gradient stop at 0%' });
    await expect.element(startStop).toBeVisible();
    await expect.element(page.getByRole('button', { exact: true, name: 'Gradient stop at 100%' })).toBeVisible();
    await act(() => (startStop.element() as HTMLElement).focus());
    await act(() => userEvent.keyboard('{ArrowRight}'));
    await settle();
    await expect.element(page.getByRole('button', { exact: true, name: 'Gradient stop at 1%' })).toBeVisible();
  });

  it('keeps the Position row identity across move, frame and transform, with the transform footer live', async () => {
    await mount(PropertiesPane, 'layer');
    await act(() => engine!.tools.setTool('move'));
    await settle();
    const xField = page.getByRole('spinbutton', { exact: true, name: 'X' }).element();
    await act(() => engine!.tools.setTool('bbox'));
    await settle();
    expect(page.getByRole('spinbutton', { exact: true, name: 'X' }).element()).toBe(xField);
    await expect.element(page.getByRole('spinbutton', { exact: true, name: 'W' })).toBeVisible();

    await act(() => engine!.tools.setTool('transform'));
    await settle();
    expect(page.getByRole('spinbutton', { exact: true, name: 'X' }).element()).toBe(xField);
    // No session and no float: Apply/Cancel are pinned but disabled.
    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toBeDisabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Cancel' })).toBeDisabled();
  });

  it('shares the selection form between lasso and marquee with the actions cluster', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('lasso'));
    await settle();
    const freehand = page.getByRole('radio', { exact: true, name: 'Freehand' });
    await expect.element(freehand).toBeVisible();
    const modeGroup = host!.querySelector<HTMLElement>('[role="group"][aria-label="Selection mode"]')!;
    expect(modeGroup).not.toBeNull();
    await expect.element(page.getByRole('button', { exact: true, name: 'Deselect' })).toBeDisabled();

    await act(() => engine!.tools.setTool('marquee'));
    await settle();
    // The op-mode buttons keep DOM identity; the shape choices swap per tool.
    expect(host!.querySelector('[role="group"][aria-label="Selection mode"]')).toBe(modeGroup);
    await expect.element(page.getByRole('radio', { exact: true, name: 'Rectangle' })).toBeVisible();
    expect(page.getByRole('radio', { exact: true, name: 'Freehand' }).query()).toBeNull();
  });

  it('remembers a group collapse per user across remounts', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    await act(() => userEvent.click(page.getByRole('button', { exact: true, name: 'Dynamics' })));
    await expect
      .element(page.getByRole('button', { exact: true, name: 'Dynamics' }))
      .toHaveAttribute('aria-expanded', 'true');

    await act(() => root?.unmount());
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    await expect
      .element(page.getByRole('button', { exact: true, name: 'Dynamics' }))
      .toHaveAttribute('aria-expanded', 'true');
    // The override reached storage, not just the in-memory store.
    expect(JSON.parse(window.localStorage.getItem('invokeai:v7:webv2:tool-property-collapsed') ?? '{}')).toMatchObject({
      'paint-dynamics': false,
    });
  });

  it('keeps the Tool section mounted with stable geometry across every tool switch', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    const section = host!.querySelector<HTMLElement>('[role="group"][aria-label="Tool"]')!;
    const { left, top } = section.getBoundingClientRect();
    for (const tool of ['eraser', 'view', 'move', 'shape'] as const) {
      await act(() => engine!.tools.setTool(tool));
      await settle();
      expect(host!.querySelector('[role="group"][aria-label="Tool"]')).toBe(section);
      const rect = section.getBoundingClientRect();
      expect(rect.left).toBe(left);
      expect(rect.top).toBe(top);
    }
  });

  it('titles the pane with the running operation, else the selected layer or group', async () => {
    const title = () => page.getByRole('heading', { level: 2 });
    await mount(PropertiesPane);
    await expect.element(title()).toHaveTextContent('No layer selected');

    await act(() => root?.unmount());
    host?.remove();
    registry?.releaseEngine('p');
    await mount(PropertiesPane, 'group');
    await expect.element(title()).toHaveTextContent('Group: Folder');

    await act(() => root?.unmount());
    host?.remove();
    registry?.releaseEngine('p');
    await mount(PropertiesPane, 'mask');
    await expect.element(title()).toHaveTextContent('Layer: Mask');
    await act(() => operations!.start(true));
    await settle();
    await expect.element(title()).toHaveTextContent('Operation: Filter');
    // The title replaces the operation section's own header; the group keeps its accessible name.
    const section = host!.querySelector<HTMLElement>('[role="group"][aria-label="Operation"]')!;
    expect(section.textContent).not.toContain('Operation');
    await act(() => operations!.start(false));
    await settle();
    await expect.element(title()).toHaveTextContent('Layer: Mask');
  });

  it('puts a running operation first with Cancel, locks the tool rows in place and hands them focus over', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    await act(() => page.getByRole('slider', { exact: true, name: 'Brush size' }).element().focus());
    await act(() => operations!.start(true));
    await settle();
    const operationSection = host!.querySelector('[role="group"][aria-label="Operation"]')!;
    const toolSection = host!.querySelector('[role="group"][aria-label="Tool"]')!;
    expect(operationSection.compareDocumentPosition(toolSection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const cancel = page.getByRole('button', { exact: true, name: 'Cancel' });
    await expect.element(cancel).toBeEnabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toBeDisabled();
    expect(document.activeElement).toBe(cancel.element());
    expect(host!.querySelector('[role="group"][inert]')?.getAttribute('aria-label')).toBe('Tool');
    await act(() => operations!.start(false));
    await settle();
    expect(host!.querySelector('[inert]')).toBeNull();
  });
});

describe('Transform pane', () => {
  it('disables its fields with nothing selected and commits one patch per field for a selected layer', async () => {
    await mount(TransformPane);
    expect(host!.textContent).toContain('No layer selected');
    await expect.element(page.getByRole('spinbutton', { exact: true, name: 'X' })).toBeDisabled();

    await act(() => root?.unmount());
    host?.remove();
    registry?.releaseEngine('p');
    await mount(TransformPane, 'layer');
    const commit = vi.spyOn(engine!.layers, 'commitPrepared');
    const x = page.getByRole('spinbutton', { exact: true, name: 'X' });
    await expect.element(x).toBeEnabled();
    await expect.element(x).toHaveValue('10');
    await act(async () => {
      await userEvent.click(x);
      await userEvent.tab();
      await userEvent.tab();
    });
    expect(commit, 'focus and blur must not commit the rounded display over 10.4').not.toHaveBeenCalled();
    await act(async () => {
      await userEvent.fill(x, '40');
      await userEvent.keyboard('{Enter}');
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(commit.mock.calls[0]?.[1]).toMatchObject({ forward: expect.anything() });
    expect(host!.textContent).toContain('Paint');
    expect(page.getByRole('button', { exact: true, name: 'Apply' }).query()).toBeNull();
  });

  it('wraps rotation into a half turn and names a selected group instead of editing it', async () => {
    await mount(TransformPane, 'layer');
    const commit = vi.spyOn(engine!.layers, 'commitPrepared');
    const rotation = page.getByRole('spinbutton', { exact: true, name: 'Rotation' });
    await expect.element(rotation).toHaveValue('28.65');
    await act(async () => {
      await userEvent.fill(rotation, '270');
      await userEvent.keyboard('{Enter}');
    });
    expect(commit).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(commit.mock.calls[0]?.[1])).toContain(String(-Math.PI / 2));

    await act(() => root?.unmount());
    host?.remove();
    registry?.releaseEngine('p');
    await mount(TransformPane, 'group');
    expect(host!.textContent).toContain('Folder');
    expect(host!.textContent).toContain('Select a layer inside this group to transform.');
    await expect.element(page.getByRole('spinbutton', { exact: true, name: 'X' })).toBeDisabled();
  });

  it('scrubs a field from its label and commits the result once when the mouse button lifts', async () => {
    await mount(TransformPane, 'layer');
    // The scrubber locks the pointer once a real click has activated the page; the harness lock steals focus.
    vi.spyOn(Element.prototype, 'requestPointerLock').mockImplementation(() => Promise.resolve());
    const commit = vi.spyOn(engine!.layers, 'commitPrepared');
    const scrubber = host!.querySelector<HTMLElement>('[data-scope="number-input"][data-part="scrubber"]')!;
    const x = page.getByRole('spinbutton', { exact: true, name: 'X' });
    await act(() =>
      scrubber.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 100, clientY: 100 }))
    );
    for (let step = 1; step <= 5; step += 1) {
      await act(() =>
        document.dispatchEvent(
          new MouseEvent('mousemove', { bubbles: true, clientX: 100 + step * 4, clientY: 100, movementX: 4 })
        )
      );
    }
    await expect.element(x).toHaveValue('15');
    expect(commit).not.toHaveBeenCalled();
    await act(() =>
      document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, button: 0, clientX: 120, clientY: 100 }))
    );
    expect(commit).toHaveBeenCalledTimes(1);
  });
});

describe('Layer editor panes host', () => {
  it('keeps the collapsed strip reachable and expands from tab selection', async () => {
    paneHarnessLayout = { ...LAYER_EDITOR_PANE_DEFAULTS };
    await mount(LayerEditorPanesHarness);
    await act(async () => {
      await userEvent.click(page.getByRole('button', { exact: true, name: 'Collapse editor panes' }));
    });
    await settle();
    const propertiesTab = page.getByRole('tab', { exact: true, name: 'Properties' });
    await expect.element(propertiesTab).toHaveAttribute('aria-selected', 'true');
    expect((propertiesTab.element() as HTMLElement).tabIndex).toBe(0);
    expect(host!.querySelector('[role="tabpanel"]')).toBeNull();
    await act(async () => {
      await userEvent.click(propertiesTab);
    });
    await settle();
    expect(host!.querySelector('[role="tabpanel"]')).not.toBeNull();
  });

  it('preserves a running operation draft across pane tab switches', async () => {
    paneHarnessLayout = { ...LAYER_EDITOR_PANE_DEFAULTS };
    await mount(LayerEditorPanesHarness);
    await act(() => engine!.tools.setTool('brush'));
    await act(() => operations!.start(true));
    await settle();
    const draft = operations!.getFilterSessionState()!.draft;
    expect(host!.querySelector('[role="group"][aria-label="Operation"]')).not.toBeNull();

    await act(async () => {
      await userEvent.click(page.getByRole('tab', { exact: true, name: 'Overview' }));
    });
    await settle();
    await act(async () => {
      await userEvent.click(page.getByRole('tab', { exact: true, name: 'Properties' }));
    });
    await settle();

    // Drafts live engine-side; remounting the pane reads the same one back
    // and the unmount must not cancel, reset, or commit the session.
    expect(operations!.getFilterSessionState()!.draft).toBe(draft);
    expect(operations!.resetFilterOperation).not.toHaveBeenCalled();
    expect(operations!.cancelFilterOperation).not.toHaveBeenCalled();
    expect(operations!.commitFilterOperation).not.toHaveBeenCalled();
    expect(host!.querySelector('[role="group"][aria-label="Operation"]')).not.toBeNull();
    await expect.element(page.getByRole('button', { exact: true, name: 'Cancel' })).toBeEnabled();
  });

  it('collapses from the separator keyboard floor and hands focus to the expand button', async () => {
    paneHarnessLayout = { ...LAYER_EDITOR_PANE_DEFAULTS, sizePx: 140 };
    await mount(LayerEditorPanesHarness);
    const separator = host!.querySelector<HTMLElement>('[role="separator"]')!;
    await act(() => separator.focus());
    await act(() => separator.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    await settle();
    const expand = page.getByRole('button', { exact: true, name: 'Expand editor panes' });
    await expect.element(expand).toBeVisible();
    expect(document.activeElement).toBe(expand.element());
  });
});

describe('Color pane', () => {
  const canvasValues = () => harness.project!.widgetInstances.canvas!.state.values as Record<string, unknown>;

  it('shows the pair on the wheel and switches the editing target', async () => {
    await mount(ColorPane);
    await expect.element(page.getByRole('slider', { exact: true, name: 'Hue' })).toBeVisible();
    const foreground = page.getByRole('button', { exact: true, name: 'Foreground color' });
    const background = page.getByRole('button', { exact: true, name: 'Background color' });
    await expect.element(foreground).toHaveAttribute('aria-pressed', 'true');
    await act(async () => {
      await userEvent.click(background);
    });
    await expect.element(background).toHaveAttribute('aria-pressed', 'true');
    await expect.element(foreground).toHaveAttribute('aria-pressed', 'false');
  });

  it('swaps the pair from the chip row', async () => {
    await mount(ColorPane);
    await act(async () => {
      await userEvent.click(page.getByRole('button', { exact: true, name: 'Swap foreground and background colors' }));
    });
    expect(canvasValues().activeColors).toEqual({ background: '#000000', foreground: '#ffffff' });
  });

  it('reveals the Mask Tint target for a selected mask and arms it only explicitly', async () => {
    clearMaskTintTarget();
    await mount(ColorPane, 'mask');
    const tintChip = page.getByRole('button', { exact: true, name: 'Mask tint' });
    await expect.element(tintChip).toBeVisible();
    // Revealed, not switched: the pair target keeps the wheel until armed.
    await expect.element(tintChip).toHaveAttribute('aria-pressed', 'false');
    const foreground = page.getByRole('button', { exact: true, name: 'Foreground color' });
    await expect.element(foreground).toHaveAttribute('aria-pressed', 'true');
    await act(async () => {
      await userEvent.click(tintChip);
    });
    await expect.element(tintChip).toHaveAttribute('aria-pressed', 'true');
    await expect.element(foreground).toHaveAttribute('aria-pressed', 'false');
    // The armed target shows the mask's tint, not the pair.
    expect((page.getByRole('textbox', { exact: true, name: 'Hex color' }).element() as HTMLInputElement).value).toBe(
      '#e07575'
    );
    // A pair chip disarms the tint and returns to the pair.
    await act(async () => {
      await userEvent.click(foreground);
    });
    await expect.element(tintChip).toHaveAttribute('aria-pressed', 'false');
    await expect.element(foreground).toHaveAttribute('aria-pressed', 'true');
  });

  it('writes swatch picks to the active target', async () => {
    await mount(SwatchesPane);
    await act(async () => {
      await userEvent.click(page.getByRole('button', { exact: true, name: '#e07575' }));
    });
    expect(canvasValues().activeColors).toMatchObject({ foreground: '#e07575' });
  });

  it('adds the current color to the project palette and removes it by right-click', async () => {
    await mount(SwatchesPane);
    await act(async () => {
      await userEvent.click(page.getByRole('button', { exact: true, name: 'Add current color to palette' }));
    });
    expect(canvasValues().colorPalette).toEqual(['#000000']);
    const paletteSwatch = page.getByRole('button', { exact: true, name: '#000000' }).last();
    await act(() =>
      paletteSwatch.element().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }))
    );
    await settle();
    expect(canvasValues().colorPalette).toEqual([]);
  });
});

const OverviewHarness = () => (
  <Box h="260px" w="300px">
    <OverviewPane />
  </Box>
);

const TREE_STRIP_TABS = [
  { id: 'layers', label: 'Layers' },
  { id: 'history', label: 'History' },
];
const TREE_STRIP_TRAILING = <span>trailing-slot</span>;

const TreeStripHarness = () => {
  const [tab, setTab] = useState('layers');
  return (
    <>
      <SegmentTabs
        activeId={tab}
        ariaLabel="Layer panel views"
        idBase="layer-tree"
        tabs={TREE_STRIP_TABS}
        trailing={TREE_STRIP_TRAILING}
        onSelect={setTab}
      />
      <Box aria-labelledby={segmentTabsTabId('layer-tree', tab)} id={segmentTabsPanelId('layer-tree')} role="tabpanel">
        {tab === 'history' ? 'history-panel' : 'layers-panel'}
      </Box>
    </>
  );
};

describe('Segment tab strip', () => {
  it('switches the middle region between the tree and history', async () => {
    await mount(TreeStripHarness);
    const layersTab = page.getByRole('tab', { exact: true, name: 'Layers' });
    const historyTab = page.getByRole('tab', { exact: true, name: 'History' });
    await expect.element(layersTab).toHaveAttribute('aria-selected', 'true');
    expect(host!.textContent).toContain('layers-panel');
    expect(host!.textContent).toContain('trailing-slot');
    await act(async () => {
      await userEvent.click(historyTab);
    });
    await expect.element(historyTab).toHaveAttribute('aria-selected', 'true');
    expect(host!.textContent).toContain('history-panel');
  });
});

describe('History pane', () => {
  it('shows the empty state with undo and redo disabled', async () => {
    await mount(HistoryPane);
    await expect.element(page.getByText('No edits yet.', { exact: true })).toBeVisible();
    await expect.element(page.getByRole('button', { exact: true, name: 'Undo canvas edit' })).toBeDisabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Redo canvas edit' })).toBeDisabled();
  });
});

describe('Overview pane', () => {
  it('draws the document composite and pans the viewport from a click', async () => {
    await mount(OverviewHarness, 'layer');
    const pan = page.getByRole('button', { exact: true, name: 'Pan the canvas view' });
    await expect.element(pan).toBeVisible();
    const canvasWidth = await pan.element().querySelector('canvas')!.width;
    expect(canvasWidth).toBeGreaterThan(0);
    const viewport = (
      harness.engine as { viewport: { getViewport: () => { getState: () => { pan: { x: number; y: number } } } } }
    ).viewport.getViewport();
    const before = viewport.getState().pan;
    await act(async () => {
      await userEvent.click(pan);
    });
    await settle();
    const after = viewport.getState().pan;
    expect(after.x === before.x && after.y === before.y).toBe(false);
  });
});
