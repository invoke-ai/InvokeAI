/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasLayerContract, CanvasLayerSourceContract, ColorSamplePhase } from '@workbench/canvas-engine/api';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type * as workbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { layerContract, stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

/** The real workbench store the hooks under test read and write, standing in for the provider. */
const harness = vi.hoisted(() => ({ store: null as WorkbenchInternalStore | null }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => {
  const actual = await importOriginal<typeof workbenchContext>();
  const { useExternalStoreSelector, shallowEqual } = await import('@platform/state/selectors');
  const store = () => harness.store!;
  return {
    ...actual,
    useActiveProjectSelector: <T,>(selector: (project: Project) => T, isEqual?: (a: T, b: T) => boolean) =>
      useExternalStoreSelector(
        store().subscribe,
        store().getSnapshot,
        (snapshot) => selector(snapshot.activeProject),
        isEqual ?? shallowEqual
      ),
    useWorkbenchCommands: () => store().commands,
    useWorkbenchQueries: () => store().queries,
  };
});

import { CanvasColorFeed } from './CanvasColorFeed';
import { armMaskTintTarget, clearMaskTintTarget } from './maskTintTarget';
import { useMaskTintEditor, type MaskTintEditor } from './useMaskTintEditor';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

const TINT = '#e07575';
const PICKED = '#3a7bd5';
const SHAPE_SOURCE = {
  fill: '#ff0000',
  height: 10,
  kind: 'rect',
  stroke: '#00ff00',
  strokeWidth: 2,
  type: 'shape',
  width: 10,
} satisfies Extract<CanvasLayerSourceContract, { type: 'shape' }>;
const TEXT_SOURCE = {
  align: 'left',
  color: '#123456',
  content: 'Hello',
  fontFamily: 'system-ui, sans-serif',
  fontSize: 48,
  fontWeight: 400,
  lineHeight: 1.2,
  type: 'text',
} satisfies Extract<CanvasLayerSourceContract, { type: 'text' }>;

type Router = (hex: string, phase: ColorSamplePhase) => boolean;

/** The Color pane's tint editor, mounted beside the feed as the pane mounts it beside the canvas. */
const TintEditorProbe = ({
  engine,
  onEditor,
}: {
  engine: CanvasEngine;
  onEditor: (editor: MaskTintEditor | null) => void;
}) => {
  onEditor(useMaskTintEditor(engine));
  return null;
};

const wait = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) {
    await cleanup();
  }
  clearMaskTintTarget();
  harness.store = null;
});

/**
 * A real engine over a project holding a mask, a shape, a text layer and a paint layer, with `selected` selected. The feed's registered router is captured so tests drive picks the way the color-picker tool does.
 */
const setup = async (selected: 'm0' | 'shape' | 'text' | 'paint', { withTintEditor = false } = {}) => {
  const store = createWorkbenchStore();
  harness.store = store;
  const projectId = store.getState().activeProjectId;
  const layers: CanvasLayerContract[] = [
    layerContract('shape', 'raster', { name: 'Shape', source: SHAPE_SOURCE }),
    layerContract('text', 'raster', { name: 'Text', source: TEXT_SOURCE }),
    layerContract('paint', 'raster', { name: 'Paint' }),
    layerContract('m0', 'inpaint_mask', { name: 'Mask' }),
  ];
  store.commands.canvas.apply(projectId, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId: selected, stacks: stacksFrom(layers) },
    type: 'replaceCanvasDocument',
  });
  const createEngine = (): CanvasEngine => {
    const created = createCanvasEngine({
      ensureProjectOnServer: () => Promise.resolve(),
      imageResolver: () => Promise.resolve(new Blob()),
      mutationPort: createCanvasProjectMutationPort(store, projectId),
      projectId,
      reportError: () => undefined,
    });
    cleanups.push(() => created.lifecycle.dispose());
    return created;
  };
  const engine = createEngine();
  const [screen, overlay] = [document.createElement('canvas'), document.createElement('canvas')];
  for (const canvas of [screen, overlay]) {
    canvas.width = 64;
    canvas.height = 64;
    document.body.append(canvas);
    cleanups.push(() => canvas.remove());
  }
  engine.surface.attach(screen, overlay);

  let router: Router | null = null;
  const setRouter = engine.tools.setColorSampleRouter.bind(engine.tools);
  vi.spyOn(engine.tools, 'setColorSampleRouter').mockImplementation((next) => {
    router = next;
    return setRouter(next);
  });

  const host = document.createElement('div');
  document.body.append(host);
  const root: Root = createRoot(host);
  cleanups.push(async () => {
    await act(() => root.unmount());
    host.remove();
  });
  let tintEditor: MaskTintEditor | null = null;
  const renderFeed = (feedEngine: CanvasEngine) =>
    act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <CanvasColorFeed engine={feedEngine} />
          {withTintEditor ? <TintEditorProbe engine={feedEngine} onEditor={(editor) => (tintEditor = editor)} /> : null}
        </I18nextProvider>
      );
    });
  await renderFeed(engine);
  expect(router).not.toBeNull();

  const document_ = () => store.getState().projects.find((project) => project.id === projectId)!.canvas.document;
  const layer = (id: string) => engine.document.model()!.getLayer(id)!;
  const canvasValues = () =>
    getProjectWidgetValues(
      store.getState().projects.find((project) => project.id === projectId)!,
      'canvas'
    );
  /** One pick: previews each color, then ends with `end` carrying the last one, as the tool does. */
  const pick = (colors: readonly string[], end: 'commit' | 'cancel') =>
    act(() => {
      for (const hex of colors) {
        router!(hex, 'preview');
      }
      router!(colors.at(-1)!, end);
    });
  return {
    canvasValues,
    document: document_,
    engine,
    layer,
    createEngine,
    pick,
    projectId,
    renderFeed,
    router: () => router!,
    store,
    tintEditor: () => tintEditor,
  };
};

const tintOf = (layer: CanvasLayerContract) => (layer.type === 'inpaint_mask' ? layer.mask.fill.color : null);
const sourceOf = (layer: CanvasLayerContract) => (layer.type === 'raster' ? layer.source : null);

describe('CanvasColorFeed Alt-pick routing', () => {
  it('writes the armed mask tint as one undoable edit and leaves the pair alone', async () => {
    const view = await setup('m0');
    armMaskTintTarget('m0');
    const pairBefore = view.canvasValues().activeColors;

    await act(() => view.router()('#101010', 'preview'));
    // A preview shows live, on the next frame, without recording history.
    await expect.poll(() => tintOf(view.layer('m0'))).toBe('#101010');
    expect(view.engine.history.getEntries().past).toEqual([]);

    await act(() => {
      view.router()(PICKED, 'preview');
      view.router()(PICKED, 'commit');
    });
    expect(tintOf(view.layer('m0'))).toBe(PICKED);
    expect(view.engine.history.getEntries().past).toEqual(['Fill']);
    expect(view.canvasValues().activeColors).toEqual(pairBefore);

    await act(() => view.engine.history.undo());
    expect(tintOf(view.layer('m0'))).toBe(TINT);
  });

  it('restores the tint from before the pick on cancel', async () => {
    const view = await setup('m0');
    armMaskTintTarget('m0');
    await view.pick(['#101010', PICKED], 'cancel');
    expect(tintOf(view.layer('m0'))).toBe(TINT);
    expect(view.engine.history.getEntries().past).toEqual([]);
  });

  it('records a pending Color pane tint edit before an Alt-pick, so the pick lands last as its own step', async () => {
    const view = await setup('m0', { withTintEditor: true });
    await act(() => armMaskTintTarget('m0'));
    // A pane edit waits out its settle delay before recording; the pick arrives inside that window.
    await act(() => view.tintEditor()!.commit('#111111'));
    await view.pick([PICKED], 'commit');
    expect(tintOf(view.layer('m0'))).toBe(PICKED);
    expect(view.engine.history.getEntries().past).toEqual(['Fill', 'Fill']);

    // The pane's settle delay passing changes nothing more.
    await act(() => wait(500));
    expect(tintOf(view.layer('m0'))).toBe(PICKED);
    expect(view.engine.history.getEntries().past).toEqual(['Fill', 'Fill']);

    await act(() => view.engine.history.undo());
    expect(tintOf(view.layer('m0'))).toBe('#111111');
  });

  it('keeps the pick on its target when the selection moves mid-pick', async () => {
    const view = await setup('m0');
    armMaskTintTarget('m0');
    await act(() => view.router()('#101010', 'preview'));
    await act(() => {
      view.store.commands.canvas.apply(view.projectId, { id: 'paint', type: 'setCanvasSelectedLayer' });
    });
    await act(() => view.router()(PICKED, 'commit'));
    expect(tintOf(view.layer('m0'))).toBe(PICKED);
    expect(view.canvasValues().activeColors).toBeUndefined();
  });

  it.each([
    ['foreground', 'fill'],
    ['background', 'stroke'],
  ] as const)('writes a selected shape’s %s-target color to its %s in one undo step', async (target, field) => {
    const view = await setup('shape');
    await act(() => view.store.commands.widgets.patchValues('canvas', { activeColorTarget: target }));
    const pairBefore = view.canvasValues().activeColors;

    await view.pick(['#101010', PICKED], 'commit');
    expect(sourceOf(view.layer('shape'))).toMatchObject({ ...SHAPE_SOURCE, [field]: PICKED });
    expect(view.engine.history.getEntries().past).toEqual(['Edit shape']);
    expect(view.canvasValues().activeColors).toEqual(pairBefore);

    await act(() => view.engine.history.undo());
    expect(sourceOf(view.layer('shape'))).toEqual(SHAPE_SOURCE);
  });

  it('writes a selected text layer’s color in one undo step', async () => {
    const view = await setup('text');
    await view.pick([PICKED], 'commit');
    expect(sourceOf(view.layer('text'))).toMatchObject({ color: PICKED });
    expect(view.engine.history.getEntries().past).toEqual(['Edit text']);
  });

  it('writes the pair, with no history, when nothing is armed — the Gradient tool included', async () => {
    const view = await setup('paint');
    await act(() => view.engine.tools.setTool('gradient'));
    await act(() => view.store.commands.widgets.patchValues('canvas', { activeColorTarget: 'background' }));

    await act(() => view.router()('#101010', 'preview'));
    // The pair follows every preview, as before.
    expect(view.canvasValues().activeColors).toEqual({ background: '#101010', foreground: '#000000' });
    await act(() => view.router()(PICKED, 'commit'));
    expect(view.canvasValues().activeColors).toEqual({ background: PICKED, foreground: '#000000' });
    expect(view.engine.history.getEntries().past).toEqual([]);
  });

  it('restores the pair color from before the pick on cancel', async () => {
    const view = await setup('paint');
    await view.pick(['#101010', PICKED], 'cancel');
    expect(view.canvasValues().activeColors).toEqual({ background: '#ffffff', foreground: '#000000' });
  });

  it('registers its router again with a new engine instance for the same project', async () => {
    const view = await setup('paint');
    const replacement = view.createEngine();
    const registered: Router[] = [];
    const setRouter = replacement.tools.setColorSampleRouter.bind(replacement.tools);
    vi.spyOn(replacement.tools, 'setColorSampleRouter').mockImplementation((next) => {
      registered.push(next);
      return setRouter(next);
    });
    expect(replacement.projectId).toBe(view.engine.projectId);

    await view.renderFeed(replacement);

    expect(registered).toHaveLength(1);
    await act(() => registered[0]!('#101010', 'preview'));
    expect(view.canvasValues().activeColors).toEqual({ background: '#ffffff', foreground: '#101010' });
  });
});
