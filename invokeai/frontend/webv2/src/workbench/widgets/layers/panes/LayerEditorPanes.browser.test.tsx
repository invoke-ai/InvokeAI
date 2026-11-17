/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type * as fonts from '@features/fonts';
import type {
  CanvasDocumentContractV3,
  CanvasLayerSourceContract,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';
import type * as CanvasOperationsApi from '@workbench/canvas-operations/api';
import type { CanvasOperationState, SamSessionSnapshot } from '@workbench/canvas-operations/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { FilterOperationSessionState } from '@workbench/canvas-operations/filterOperationSession';
import type { CanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import type { Project } from '@workbench/projectContracts';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createDocumentModel, getDocumentLayer, getDocumentLeaves } from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { IDLE_CONTROL_DETECTION_STATUS } from '@workbench/canvas-operations/api';
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
  appliesMutations: false,
  engine: null as unknown,
  listeners: new Set<() => void>(),
  project: null as Project | null,
}));

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
  const { applyCanvasProjectMutation } = await import('@workbench/canvasProjectMutations');
  const applyCanvas = (projectId: string, mutation: Parameters<typeof applyCanvasProjectMutation>[1]) => {
    const project = harness.project;
    if (!project || project.id !== projectId) {
      return false;
    }
    harness.project = applyCanvasProjectMutation(project, mutation);
    harness.listeners.forEach((listener) => listener());
    return harness.project.canvas !== project.canvas;
  };
  const commands = { canvas: { apply: applyCanvas }, widgets: { patchValues } };
  const queries = {
    getProject: (projectId: string) => (harness.project?.id === projectId ? harness.project : undefined),
    getSnapshot: () => ({ activeProject: harness.project! }),
  };
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
vi.mock('@workbench/canvas-operations/api', async (importOriginal) => ({
  ...(await importOriginal<typeof CanvasOperationsApi>()),
  getCanvasEngine: () => harness.engine,
}));

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

const createEngineDeps = (): EngineDeps => {
  const mutationPort: CanvasProjectMutationPort = {
    commitEdit: () => undefined,
    dispatch: (mutation) => {
      if (!harness.appliesMutations || !harness.project) {
        return false;
      }
      const before = harness.project.canvas;
      harness.project = applyCanvasProjectMutation(harness.project, mutation);
      harness.listeners.forEach((listener) => listener());
      return harness.project.canvas !== before;
    },
    getCanvasState: () => harness.project?.canvas ?? createEmptyCanvasState(64, 64),
    subscribe: (listener) => {
      harness.listeners.add(listener);
      return () => harness.listeners.delete(listener);
    },
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

const samSession = (patch: Partial<SamSessionSnapshot> = {}): SamSessionSnapshot => ({
  applyPolygonRefinement: false,
  autoProcess: true,
  error: null,
  hasPreview: false,
  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
  invert: false,
  isolatedPreview: true,
  layerName: 'Portrait',
  layerType: 'raster',
  model: 'segment-anything-2-large',
  pointLabel: 'include',
  retrying: false,
  sourceRect: { height: 768, width: 1024, x: 0, y: 0 },
  status: 'ready',
  ...patch,
});

const createFakeOperations = () => {
  const listeners = new Set<() => void>();
  const subscribe = (listener: () => void) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };
  let filter: FilterOperationSessionState | null = null;
  let sam: SamSessionSnapshot | null = null;
  let state: CanvasOperationState = { status: 'idle' };
  return {
    applySelectObjectSession: vi.fn(),
    cancelFilterOperation: vi.fn(),
    cancelSelectObjectSession: vi.fn(),
    commitFilterOperation: vi.fn(),
    getFilterSessionState: () => filter,
    getOperationState: () => state,
    getSamSessionState: () => sam,
    processFilterOperation: vi.fn(),
    processSelectObjectSession: vi.fn(() => Promise.resolve('completed' as const)),
    resetFilterOperation: vi.fn(),
    resetSelectObjectSession: vi.fn(() => 'updated' as const),
    saveSelectObjectSession: vi.fn(),
    setFilterOperationAutoProcess: vi.fn(),
    startSam: (patch: Partial<SamSessionSnapshot> = {}) => {
      sam = samSession(patch);
      state = {
        error: null,
        identity: { kind: 'select-object', layerId: 'layer-1', projectId: 'p' },
        phase: 'ready',
        status: 'active',
      };
      listeners.forEach((listener) => listener());
    },
    setSam: (patch: Partial<SamSessionSnapshot>) => {
      sam = { ...sam!, ...patch };
      listeners.forEach((listener) => listener());
    },
    updateSelectObjectSession: vi.fn(() => 'updated' as const),
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
    getControlDetectionStatus: () => IDLE_CONTROL_DETECTION_STATUS,
    subscribeControlDetectionStatus: () => () => undefined,
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

type Selection = 'none' | 'layer' | 'group' | 'mask' | 'control' | 'shape' | 'stroked-shape' | 'gradient' | 'text';

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

const mount = async (
  View: typeof PropertiesPane,
  selection: Selection = 'none',
  { extraNodes = [], width = 450 }: { extraNodes?: Parameters<typeof stacksFrom>[0]; width?: number } = {}
) => {
  const base = { ...createInitialWorkbenchState().projects[0]!, id: 'p' };
  const nodes =
    selection === 'none'
      ? []
      : selection === 'layer'
        ? [layerContract('l0', 'raster', { name: 'Paint', transform: { ...IDENTITY, rotation: 0.5, x: 10.4 } })]
        : selection === 'shape' || selection === 'stroked-shape'
          ? [
              layerContract('l0', 'raster', {
                name: 'My Shape',
                source: selection === 'shape' ? SHAPE_SOURCE : { ...SHAPE_SOURCE, stroke: '#00ff00', strokeWidth: 4 },
              }),
            ]
          : selection === 'gradient'
            ? [layerContract('l0', 'raster', { name: 'My Gradient', source: GRADIENT_SOURCE })]
            : selection === 'text'
              ? [layerContract('l0', 'raster', { name: 'My Text', source: TEXT_SOURCE })]
              : selection === 'mask'
                ? [layerContract('m0', 'inpaint_mask', { name: 'Mask' })]
                : selection === 'control'
                  ? [layerContract('l0', 'control', { name: 'Control Layer 1' })]
                  : [groupContract('g0', [layerContract('l0', 'raster', { name: 'Paint' })], { name: 'Folder' })];
  harness.project = applyCanvasProjectMutation(base, {
    document: {
      ...createEmptyCanvasDocument(),
      selectedLayerId: selection === 'none' ? null : selection === 'mask' ? 'm0' : selection === 'group' ? 'g0' : 'l0',
      stacks: stacksFrom([...nodes, ...extraNodes]),
    },
    type: 'replaceCanvasDocument',
  });
  host = document.createElement('div');
  host.style.cssText = `width:${width}px;height:600px;`;
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
  harness.appliesMutations = false;
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
  // Collapse overrides persist in localStorage, which the browser session shares across test FILES; reset around every
  // test so no order dependence leaks in either direction.
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
    const dynamics = page.getByRole('button', { exact: true, name: 'Dynamics' });
    await expect.element(dynamics).toHaveAttribute('aria-expanded', 'false');
    expect(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects width' }).query()).toBeNull();
    await act(() => userEvent.click(dynamics));
    await expect.element(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects width' })).toBeVisible();
    await expect
      .element(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects opacity' }))
      .toBeVisible();

    await act(() => engine!.tools.setTool('eraser'));
    await settle();
    const eraserWidth = page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects width' });
    await expect.element(eraserWidth).toBeChecked();
    expect(page.getByRole('checkbox', { exact: true, name: 'Pen pressure affects opacity' }).query()).toBeNull();
    await act(() => userEvent.click(page.getByText('Pen pressure affects width', { exact: true })));
    expect(engine!.interaction.get('eraserOptions').pressureAffectsWidth).toBe(false);
    expect(engine!.interaction.get('brushOptions').pressureAffectsWidth).toBe(true);

    await act(() => engine!.tools.setTool('view'));
    await settle();
    expect(page.getByRole('slider', { exact: true, name: 'Brush size' }).query()).toBeNull();
    expect(host!.textContent).not.toContain('Pan the canvas.');
    expect(host!.textContent).not.toContain('Hold Space');
    expect(host!.textContent).not.toContain('Tool');
    expect(host!.textContent).toContain('Layer');
  });

  it('edits only the selected shape appearance, with one undo step per gesture', async () => {
    harness.appliesMutations = true;
    await mount(PropertiesPane, 'stroked-shape');
    await act(() => engine!.tools.setTool('view'));
    const optionsBefore = engine!.interaction.get('shapeOptions');
    const colorsBefore = harness.project!.widgetInstances.canvas!.state.values;
    const layer = page.getByRole('group', { exact: true, name: 'Layer' });
    const source = () => {
      const selected = getDocumentLayer(harness.project!.canvas.document, 'l0');
      if (selected?.type !== 'raster' || selected.source.type !== 'shape') {
        throw new Error('Expected selected shape');
      }
      return selected.source;
    };
    const toggle = (name: string) =>
      act(() => layer.element().querySelector<HTMLElement>(`button[aria-label="${name}"][aria-pressed]`)!.click());

    await toggle('Fill');
    expect(source().fill).toBeNull();
    expect(engine!.history.getEntries().past).toHaveLength(1);
    const width = layer.getByRole('slider', { exact: true, name: 'Stroke width' });
    await act(async () => {
      (width.element() as HTMLElement).focus();
      await userEvent.keyboard('{ArrowRight>3/}');
    });
    expect(source().strokeWidth).toBe(7);
    expect(engine!.history.getEntries().past).toHaveLength(2);
    await act(() => engine!.history.undo());
    expect(source().strokeWidth).toBe(4);
    expect(source().fill).toBeNull();
    await toggle('Stroke');
    expect(source().stroke).toBeNull();
    await expect.element(width).toHaveAttribute('aria-disabled', 'true');
    expect(source().kind).toBe('rect');
    expect(source().width).toBe(10);
    expect(engine!.interaction.get('shapeOptions')).toBe(optionsBefore);
    expect(harness.project!.widgetInstances.canvas!.state.values).toBe(colorsBefore);

    await act(() => operations!.start(true));
    expect(layer.element().hasAttribute('inert')).toBe(true);
    await act(() => operations!.start(false));
    expect(layer.element().hasAttribute('inert')).toBe(false);
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
    expect(JSON.parse(window.localStorage.getItem('invokeai:v7:webv2:tool-property-collapsed') ?? '{}')).toMatchObject({
      'paint-dynamics': false,
    });
  });

  it('puts a running operation first with Cancel, locks the tool rows in place and hands them focus over', async () => {
    await mount(PropertiesPane);
    await act(() => engine!.tools.setTool('brush'));
    await settle();
    const slider = page.getByRole('slider', { exact: true, name: 'Brush size' }).element();
    await act(() => slider.focus());
    const viewport = [...host!.querySelectorAll<HTMLElement>('*')].find(
      (element) => element.scrollHeight > element.clientHeight && getComputedStyle(element).overflowY !== 'visible'
    );
    if (viewport) {
      viewport.scrollTop = viewport.scrollHeight;
    }
    await act(() => operations!.start(true));
    await settle();
    const operationSection = host!.querySelector('[role="group"][aria-label="Operation"]')!;
    const toolSection = host!.querySelector('[role="group"][aria-label="Tool"]')!;
    expect(operationSection.compareDocumentPosition(toolSection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const cancel = page.getByRole('button', { exact: true, name: 'Cancel' });
    await expect.element(cancel).toBeEnabled();
    await expect.element(page.getByRole('button', { exact: true, name: 'Apply' })).toBeDisabled();
    expect(document.activeElement).toBe(cancel.element());
    const hostRect = host!.getBoundingClientRect();
    const cancelRect = cancel.element().getBoundingClientRect();
    expect(cancelRect.top).toBeGreaterThanOrEqual(hostRect.top);
    expect(cancelRect.bottom).toBeLessThanOrEqual(hostRect.bottom);
    expect(host!.querySelector('[role="group"][inert]')?.getAttribute('aria-label')).toBe('Tool');
    await act(() => operations!.start(false));
    await settle();
    expect(host!.querySelector('[inert]')).toBeNull();
  });
});

describe('Select Object properties', () => {
  beforeEach(() => resetPropertyGroupCollapse());
  const startSam = async (patch: Partial<SamSessionSnapshot> = {}, width?: number) => {
    await mount(PropertiesPane, 'none', width ? { width } : undefined);
    await act(() => operations!.startSam(patch));
    await settle();
  };

  it('lays points, box and settings on pane rows and switches input type by intent', async () => {
    await startSam();

    await expect.element(page.getByRole('radiogroup', { name: 'Select by' })).toBeVisible();
    await expect.element(page.getByRole('radio', { name: 'Include 0' })).toBeChecked();
    await expect.element(page.getByText('Drag on the canvas')).toBeVisible();
    expect(page.getByRole('radiogroup', { name: 'Points' }).element().getAttribute('aria-describedby')).toBeTruthy();
    await userEvent.click(page.getByText('Settings'));
    await expect.element(page.getByRole('combobox', { name: 'Model' })).toBeVisible();
    for (const name of ['Refine edges', 'Auto-update preview', 'Show only the object']) {
      await expect.element(page.getByRole('checkbox', { name })).toBeVisible();
    }

    await userEvent.click(page.getByText('Prompt', { exact: true }));
    expect(operations!.updateSelectObjectSession).toHaveBeenCalledExactlyOnceWith({ inputType: 'prompt' });
  });

  it('finds the prompted object on Enter whatever auto-update says, and offers Find object while manual', async () => {
    await startSam({ input: { prompt: 'cat', type: 'prompt' } });
    const prompt = page.getByRole('textbox', { name: 'Prompt' });
    expect(page.getByRole('button', { name: 'Find object' }).query()).toBeNull();

    await userEvent.click(prompt);
    await userEvent.keyboard('{Enter}');
    expect(operations!.processSelectObjectSession).toHaveBeenCalledOnce();

    await act(() => operations!.setSam({ autoProcess: false }));
    await userEvent.click(page.getByRole('button', { name: 'Find object' }));
    expect(operations!.processSelectObjectSession).toHaveBeenCalledTimes(2);
  });
});

describe('Tool property scrubbers', () => {
  beforeEach(() => resetPropertyGroupCollapse());
  afterEach(() => resetPropertyGroupCollapse());

  const scrubber = async (name: string, section: 'Tool' | 'Layer' = 'Tool') => {
    const slider = page.getByRole('group', { exact: true, name: section }).getByRole('slider', { exact: true, name });

    await expect.element(slider).toBeInTheDocument();
    return slider.element() as HTMLElement;
  };

  const drag = async (slider: HTMLElement, from: number, ...to: number[]) => {
    const frame = slider.closest<HTMLElement>('[data-scope="scrubber"]')!;
    const { left, width } = frame.getBoundingClientRect();
    let last = from;
    const fire = (target: EventTarget, type: string, fraction: number) => {
      last = fraction;
      return act(() =>
        target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: left + width * fraction }))
      );
    };
    const move = async (...fractions: number[]) => {
      for (const fraction of fractions) {
        await fire(window, 'pointermove', fraction);
      }
    };
    await fire(frame, 'pointerdown', from);
    await move(...to);
    return { move, release: () => fire(window, 'pointerup', last) };
  };

  const holdKey = async (slider: HTMLElement, key: string, repeats: number) => {
    for (let index = 0; index < repeats; index += 1) {
      await act(() => slider.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key, repeat: index > 0 })));
    }
    return () => act(() => slider.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true, key })));
  };

  const observe = (optionsKey: 'gradientOptions' | 'shapeOptions' | 'textOptions') => {
    let live: CanvasDocumentContractV3 = harness.project!.canvas.document;
    vi.spyOn(engine!.document, 'model').mockImplementation(() =>
      createDocumentModel(live, { editRevision: 0, projectId: 'p' })
    );
    const commits: PreparedDocumentEdit[] = [];
    vi.spyOn(engine!.layers, 'commitPrepared').mockImplementation((_label, edit) => {
      commits.push(edit);
      return { status: 'committed' };
    });
    const set = vi.spyOn(engine!.interaction, 'set');
    return {
      commits,
      committedSources: () =>
        commits.map(
          (edit) => (edit.forward as { source: CanvasLayerSourceContract }).source as Record<string, unknown>
        ),
      previews: () => set.mock.calls.filter(([key]) => key === optionsKey).length,
      setLiveSource: (source: CanvasLayerSourceContract) => {
        live = applyCanvasProjectMutation(harness.project!, { id: 'l0', source, type: 'updateCanvasLayerSource' })
          .canvas.document;
      },
    };
  };

  it.each([
    ['text', 'Font size', 'textOptions'],
    ['stroked-shape', 'Stroke width', 'shapeOptions'],
  ] as const)(
    'shows draft steps and commits once per drag or held key (%s, %s)',
    async (selection, name, optionsKey) => {
      await mount(PropertiesPane, selection);
      await act(() => engine!.tools.setTool(selection === 'stroked-shape' ? 'shape' : selection));
      await settle();
      const { commits, previews } = observe(optionsKey);
      const slider = await scrubber(name, selection === 'stroked-shape' ? 'Layer' : 'Tool');

      const gesture = await drag(slider, 0.5, 0.55, 0.6);
      expect(previews()).toBe(selection === 'stroked-shape' ? 0 : 2);
      expect(commits).toHaveLength(0);
      await gesture.release();
      expect(commits).toHaveLength(1);

      const settled = previews();
      const releaseKey = await holdKey(slider, 'ArrowRight', 3);
      expect(previews() - settled).toBe(selection === 'stroked-shape' ? 0 : 3);
      expect(commits).toHaveLength(1);
      await releaseKey();
      expect(commits).toHaveLength(2);
    }
  );

  it('keeps an undone colour in the font-size commit when the undo lands mid-drag', async () => {
    await mount(PropertiesPane, 'text');
    await act(() => engine!.tools.setTool('text'));
    await settle();
    const { committedSources, setLiveSource } = observe('textOptions');

    const { release } = await drag(await scrubber('Font size'), 0.5, 0.6);
    setLiveSource({ ...TEXT_SOURCE, color: '#abcdef' });
    await release();

    const [committed] = committedSources();
    expect(committed?.color).toBe('#abcdef');
    expect(committed?.fontSize).toBeGreaterThan(TEXT_SOURCE.fontSize);
    expect(committed?.content).toBe(TEXT_SOURCE.content);
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
});

describe('Layer editor panes host', () => {
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

describe('History pane', () => {
  const steps = () =>
    [...page.getByRole('list', { name: 'Edit history' }).element().querySelectorAll('button')].map(
      (button) => button.textContent
    );

  it('jumps back to the start, and clears history only once that is confirmed', async () => {
    harness.appliesMutations = true;
    await mount(HistoryPane, 'layer');
    const bbox = harness.project!.canvas.document.bbox;
    await act(() => {
      engine!.layers.commitStructural(
        'Move frame',
        { bbox: { ...bbox, x: 8 }, type: 'setCanvasBbox' },
        {
          bbox,
          type: 'setCanvasBbox',
        }
      );
    });
    expect(steps()).toEqual(['Start', 'Move frame']);

    await act(() => userEvent.click(page.getByRole('button', { exact: true, name: 'Start' })));
    await expect.poll(() => engine!.history.getEntries().future).toEqual(['Move frame']);

    await act(() => userEvent.click(page.getByRole('button', { exact: true, name: 'Clear history' })));
    const dialog = page.getByRole('alertdialog');
    await expect.element(dialog).toBeVisible();
    expect(dialog.element().textContent).toContain('Discards the 1 undo step for this canvas.');
    await act(() => userEvent.click(dialog.getByRole('button', { exact: true, name: 'Clear history' })));
    await expect.poll(() => engine!.history.getEntries().future).toEqual([]);
    expect(steps()).toEqual(['Start']);
  });

  it('restores a canvas point, keeping the canvas it replaced as a point in its place', async () => {
    harness.appliesMutations = true;
    await mount(HistoryPane, 'layer');
    const document = {
      ...harness.project!.canvas.document,
      selectedLayerId: 'restored',
      stacks: stacksFrom([layerContract('restored', 'raster', { name: 'Restored' })]),
    };
    await act(() => {
      harness.project = {
        ...harness.project!,
        canvas: {
          ...harness.project!.canvas,
          snapshots: [
            { createdAt: new Date().toISOString(), document, id: 'a', kind: 'clear-canvas', name: 'Earlier' },
          ],
        },
      };
      harness.listeners.forEach((listener) => listener());
    });

    await act(() =>
      userEvent.click(page.getByRole('list', { name: 'Restore points' }).getByRole('button', { name: 'Restore' }))
    );
    await expect
      .poll(() => getDocumentLeaves(harness.project!.canvas.document).map((layer) => layer.id))
      .toEqual(['restored']);
    const kept = harness.project!.canvas.snapshots;
    expect(kept.map((point) => point.name)).toEqual(['Before restoring “Earlier”']);
    expect(getDocumentLeaves(kept[0]!.document).map((layer) => layer.id)).toEqual(['l0']);
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
