import type { ArchitectureCapabilitiesRow } from '@features/models/architectures';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasLayerPreviewMutation,
} from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import type { ControlLayerIntent, ControlLayerIntentResult } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { ensureArchitectureCapabilitiesLoaded } from '@features/models/architecturesStore';
import { architectureCapabilitiesFixture } from '@features/models/core/architectureCapabilities.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { createControlLayer, createDocumentModel, getDocumentLayer } from '@workbench/canvas-engine/api';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { previewInverse } from '@workbench/canvas-engine/document-model/documentModel';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { normalizeControlAdapter } from '@workbench/controlAdapters';
import { resetPropertyGroupCollapse } from '@workbench/widgets/canvas/tool-presentation/propertyGroupStore';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ControlLayerSettings } from './ControlLayerSettings';

const harness = vi.hoisted(() => ({
  canManageModels: true,
  mainModel: { base: 'sdxl', key: 'sdxl-main', name: 'SDXL', type: 'main' } as Record<string, unknown> | null,
  memory: [] as [string, string | null][],
  remembered: {} as Record<string, string>,
  startersRequested: 0,
  models: [] as Record<string, unknown>[],
  navigate: [] as unknown[],
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureStartersLoaded: () => {
    harness.startersRequested += 1;
  },
  useModelsSelector: (selector: (snapshot: { models: typeof harness.models }) => unknown) =>
    selector({ models: harness.models }),
}));
vi.mock('./useSelectedMainModel', () => ({ useSelectedMainModel: () => harness.mainModel }));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useCapabilities: () => ({ canManageModels: harness.canManageModels, canManageNodes: true, canManageUsers: true }),
}));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => (options: unknown) => {
    harness.navigate.push(options);
    return Promise.resolve();
  },
}));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'test-project',
  useOptionalWorkbenchCommands: () => null,
}));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkbenchPreferenceSelector: (
    selector: (preferences: { controlModelMemory: Record<string, string> }) => unknown
  ) => selector({ controlModelMemory: harness.remembered }),
  patchControlModelMemory: (key: string, model: string | null) => {
    harness.memory.push([key, model]);
    return Promise.resolve();
  },
}));
const getArchitectureCapabilities = vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>();
vi.mock('@features/models/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));

const i18n = createInstance();
beforeAll(async () => {
  const translation = (await (await fetch('/locales/en.json')).json()) as Record<string, unknown>;
  await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = (key: string, type = 'controlnet', base = 'sdxl') => ({ base, key, name: key, source: key, type });
const DEPTH_A = model('controlnet-depth-sdxl-1.0');
const DEPTH_B = model('controlnet-zoe-depth-sdxl');
const POSE = model('OpenPose XL');

const PHOTO = { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' } as const;

const controlLayer = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => {
  const created = createControlLayer({ base: 'sdxl', id: 'control-1', name: 'Control Layer 1' });
  return { ...created, ...overrides, adapter: { ...created.adapter, ...overrides.adapter } };
};
const detectedLayer = (
  controlType: CanvasControlLayerContract['controlType'],
  detector: string,
  overrides: Partial<CanvasControlLayerContract> = {}
) =>
  controlLayer({
    controlType,
    detection: { settings: {}, type: detector },
    filter: { settings: {}, type: detector },
    original: PHOTO,
    source: PHOTO,
    ...overrides,
  });

let host: HTMLDivElement | null = null;
let root: Root | null = null;
const queryClient = new QueryClient();

const settle = async (run: () => unknown = () => undefined) => {
  await act(async () => {
    await run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

type Engine = NonNullable<Parameters<typeof ControlLayerSettings>[0]['engine']>;

const commits: { label: string; edit: unknown }[] = [];
const previews: CanvasLayerPreviewMutation[] = [];
const restores: CanvasLayerPreviewMutation[] = [];
const intents: ControlLayerIntent[] = [];
const retries: string[] = [];
let intentResult: ControlLayerIntentResult = 'updated';
const statuses = createControlDetectionStatusStore();

const engineWith = (layers: CanvasLayerContract[], { hasContent = true } = {}): Engine => {
  const document = { ...createEmptyCanvasDocument(), stacks: stacksFrom(layers) };
  const context = { editRevision: 0, projectId: 'test-project' };
  const documentModel = createDocumentModel(document, context);
  let liveModel = documentModel;
  const showPreview = (action: CanvasLayerPreviewMutation) => {
    if (action.type !== 'updateCanvasLayerConfig' || action.config.layerType !== 'control') {
      return;
    }
    const { adapter } = action.config;
    const live = liveModel.document;
    const shown = [...live.stacks.control].map((node) =>
      node.id === action.id && node.type === 'control' ? { ...node, adapter: { ...node.adapter, ...adapter } } : node
    );
    liveModel = createDocumentModel({ ...live, stacks: { ...live.stacks, control: shown } }, context);
  };
  const engine = {
    document: { getDocument: () => document, model: () => liveModel },
    exports: {
      exportRasterComposite: () => ({ status: 'empty' }),
      hasExportableLayerContent: () => hasContent,
    },
    interaction: { get: () => false, subscribe: () => () => undefined },
    layers: {
      beginStructuralPreview: () => {
        let baseline: CanvasLayerPreviewMutation | null = null;
        return {
          apply: (action: CanvasLayerPreviewMutation) => {
            baseline ??= previewInverse(documentModel.getNode(action.id)!, action);
            previews.push(action);
            showPreview(action);
            return true;
          },
          baseline: () => baseline,
          cancel: () => {
            liveModel = documentModel;
            if (baseline) {
              restores.push(baseline);
            }
          },
          commit: (label: string, edit: unknown) => {
            commits.push({ edit, label });
            return { status: 'committed' as const };
          },
          isActive: () => true,
        };
      },
      commitPrepared: (label: string, edit: unknown) => {
        commits.push({ edit, label });
        return { status: 'committed' as const };
      },
      endStructuralPreview: () => undefined,
    },
    lifecycle: { flushPendingUploads: () => Promise.resolve() },
    projectId: 'test-project',
    tools: { setTool: vi.fn() },
  } as unknown as Engine;
  attachOperations(engine);
  return engine;
};

const attachOperations = (engine: Engine): void => {
  attachCanvasOperations(engine, {
    applyControlLayerIntent: (_layerId: string, intent: ControlLayerIntent) => {
      intents.push(intent);
      return intentResult;
    },
    getControlDetectionStatus: statuses.get,
    retryControlDetection: (layerId: string) => {
      retries.push(layerId);
      return 'scheduled';
    },
    subscribeControlDetectionStatus: statuses.subscribe,
  } as never);
};

const render = async (
  layer: CanvasControlLayerContract,
  { engine = engineWith([layer]), width = 320 }: { engine?: Engine | null; width?: number } = {}
) => {
  applyThemeToRoot(DEFAULT_THEME_ID);
  host = document.createElement('div');
  host.style.width = `${width}px`;
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <ControlLayerSettings engine={engine} layer={layer} />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};
const unmount = async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
};

const loadCapabilities = async () => {
  getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
  await settle(ensureArchitectureCapabilitiesLoaded);
};

beforeEach(async () => {
  resetPropertyGroupCollapse();
  harness.models = [DEPTH_A, POSE];
  await loadCapabilities();
});

afterEach(async () => {
  await unmount();
  getArchitectureCapabilities.mockReset();
  harness.canManageModels = true;
  harness.mainModel = { base: 'sdxl', key: 'sdxl-main', name: 'SDXL', type: 'main' };
  harness.memory.length = 0;
  harness.remembered = {};
  harness.startersRequested = 0;
  harness.navigate.length = 0;
  commits.length = 0;
  previews.length = 0;
  restores.length = 0;
  intents.length = 0;
  retries.length = 0;
  intentResult = 'updated';
  statuses.clear();
  resetPropertyGroupCollapse();
  accountLifecycle.invalidate();
});
const combobox = (name: string) => page.getByRole('combobox', { exact: true, name });
const choose = async (name: string, option: string | RegExp) => {
  await act(() => combobox(name).click());
  await act(() => page.getByRole('option', { name: option }).click());
  await settle();
};
const adapterOf = (index: number) =>
  (commits[index]!.edit as { forward: { config: { adapter: Record<string, unknown> } } }).forward.config.adapter;

describe('ControlLayerSettings type and detection', () => {
  it('hints each type with what is installed, and renames an auto-named layer after the type', async () => {
    harness.models = [DEPTH_A, DEPTH_B, POSE];
    await render(controlLayer({ source: PHOTO }));

    await act(() => combobox('Type').click());
    for (const name of ['Depth, 2 models', 'Pose, 1 model', 'Line art, Not installed']) {
      await expect.element(page.getByRole('option', { exact: true, name })).toBeVisible();
    }
    await act(() => page.getByRole('option', { name: /^Depth/ }).click());
    await settle();

    expect(intents).toEqual([{ controlType: 'depth', kind: 'set-type', name: 'Depth 1' }]);
  });

  it('shows a spinner while detection waits or runs, then offers Run again with the failure reason', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    await settle(() => statuses.set('control-1', { status: 'queued' }));
    expect(combobox('Detection').element().querySelector('.chakra-spinner')).not.toBeNull();
    expect(page.getByRole('button', { name: 'Detect again' }).elements()).toHaveLength(0);

    await settle(() =>
      statuses.set('control-1', { detector: 'depth_anything_depth_estimation', status: 'processing' })
    );
    expect(combobox('Detection').element().querySelector('.chakra-spinner')).not.toBeNull();

    await settle(() => statuses.set('control-1', { error: { code: 'process-failed' }, status: 'error' }));
    expect(combobox('Detection').element().querySelector('.chakra-spinner')).toBeNull();
    await userEvent.hover(page.getByRole('button', { name: 'Run again' }));
    await expect.element(page.getByText('Depth Anything failed: the detector could not run')).toBeVisible();
    await act(() => page.getByRole('button', { name: 'Run again' }).click());

    expect(retries).toEqual(['control-1']);
  });

  it('opens the detector settings inline and sends one intent per scrub gesture', async () => {
    await render(
      detectedLayer('edges', 'canny_edge_detection', { detection: { settings: {}, type: 'canny_edge_detection' } })
    );
    const toggle = page.getByRole('button', { name: 'Detector settings' });
    await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');

    await act(() => toggle.click());
    await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
    const region = document.getElementById(toggle.element().getAttribute('aria-controls')!)!;
    expect(region.getAttribute('role')).toBe('group');

    const frame = region.querySelector<HTMLElement>('[data-scope="scrubber"]')!;
    const rect = frame.getBoundingClientRect();
    const pointer = (target: EventTarget, type: string, x: number) =>
      settle(() =>
        target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: x, pointerId: 1 }))
      );
    await pointer(frame, 'pointerdown', rect.left + 10);
    await pointer(window, 'pointermove', rect.left + 40);
    await pointer(window, 'pointermove', rect.left + 80);
    await pointer(window, 'pointerup', rect.left + 80);

    expect(intents.map((intent) => intent.kind)).toEqual(['set-detector-settings']);
  });
});

describe('ControlLayerSettings model', () => {
  it('asks for a choice among several models, and records a pick for the layer and for later layers', async () => {
    harness.models = [DEPTH_A, DEPTH_B];
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    expect(combobox('Model').element().textContent).toContain('Choose a depth model');
    expect(host!.textContent).toContain('2 models fit: controlnet-depth-sdxl-1.0 or controlnet-zoe-depth-sdxl');
    await act(() => combobox('Model').click());
    expect(page.getByRole('group', { name: 'Made for depth' }).elements()).toHaveLength(1);
    await act(() => page.getByRole('option', { name: /^controlnet-zoe-depth-sdxl/ }).click());
    await settle();

    expect(commits.map(({ label }) => label)).toEqual(['Model']);
    expect(adapterOf(0)).toEqual({ kind: 'controlnet', model: 'controlnet-zoe-depth-sdxl' });
    expect(harness.memory).toEqual([['sdxl:depth', 'controlnet-zoe-depth-sdxl']]);
  });

  it("resets the layer's own pick back to automatic and returns focus to the model choice", async () => {
    harness.models = [DEPTH_A, DEPTH_B];
    const picked = detectedLayer('depth', 'depth_anything_depth_estimation', {
      adapter: { ...controlLayer().adapter, model: 'controlnet-zoe-depth-sdxl' },
    });
    await render(picked);
    expect(host!.textContent).toContain('ControlNet · your pick for depth');

    await act(() => page.getByRole('button', { name: 'Reset' }).click());
    await settle();

    expect(adapterOf(0)).toEqual({ model: null });
    expect(harness.memory).toEqual([]);
    expect(document.activeElement).toBe(combobox('Model').element());
  });
});

describe('ControlLayerSettings adapter rows', () => {
  const weightSlider = () => page.getByRole('slider', { name: 'Weight' });

  it('records one step for a weight drag, from the weight it started at', async () => {
    const layer = detectedLayer('depth', 'depth_anything_depth_estimation', {
      adapter: { ...controlLayer().adapter, weight: 1 },
    });
    await render(layer);
    const control = weightSlider()
      .element()
      .closest<HTMLElement>('[data-part="root"]')!
      .querySelector<HTMLElement>('[data-part="control"]')!;
    const rect = control.getBoundingClientRect();
    const xAt = (weight: number) => rect.left + (rect.width * weight) / 2;
    const pointer = (target: EventTarget, type: string, x: number) =>
      settle(() =>
        target.dispatchEvent(
          new PointerEvent(type, {
            bubbles: true,
            button: 0,
            clientX: x,
            clientY: rect.top + rect.height / 2,
            pointerId: 1,
          })
        )
      );
    await pointer(control, 'pointerdown', xAt(1));
    await pointer(document, 'pointermove', xAt(1.25));
    await pointer(document, 'pointermove', xAt(1.5));
    await pointer(document, 'pointerup', xAt(1.5));

    expect(previews.length).toBeGreaterThanOrEqual(2);
    expect(commits).toHaveLength(1);
    const { forward, inverse } = commits[0]!.edit as Record<
      'forward' | 'inverse',
      { config: { adapter: { weight: number } } }
    >;
    expect(forward.config.adapter.weight).toBeCloseTo(1.5, 0);
    expect(forward.config.adapter.weight).toBe(
      (previews.at(-1) as { config: { adapter: { weight: number } } }).config.adapter.weight
    );
    expect(inverse.config.adapter.weight).toBe(1);
  });

  it('shows a stored legacy Unbalanced mode as a fourth segment the layer can leave', async () => {
    await render(
      detectedLayer('depth', 'depth_anything_depth_estimation', {
        adapter: { ...controlLayer().adapter, controlMode: 'unbalanced' },
      })
    );

    expect(page.getByRole('radio').elements()).toHaveLength(4);
    await expect.element(page.getByRole('radio', { name: 'Unbalanced' })).toBeChecked();
    await act(() => page.getByRole('radiogroup', { name: 'Priority' }).getByText('Balanced', { exact: true }).click());
    await settle();

    expect(adapterOf(0)).toEqual({ controlMode: 'balanced' });
  });

  it('sets the priority of a layer created as another kind, and a reload keeps it', async () => {
    const layer = detectedLayer('depth', 'depth_anything_depth_estimation', {
      adapter: { ...controlLayer().adapter, controlMode: null, kind: 'z_image_control', weight: -0.5 },
    });
    await render(layer);
    await act(() => page.getByRole('radiogroup', { name: 'Priority' }).getByText('Control', { exact: true }).click());
    await settle();

    expect(adapterOf(0)).toEqual({ controlMode: 'more_control' });
    expect(normalizeControlAdapter({ ...layer.adapter, ...adapterOf(0) })).toMatchObject({
      controlMode: 'more_control',
      kind: 'z_image_control',
      weight: -0.5,
    });
  });
});

describe('ControlLayerSettings adapter gestures on the live document', () => {
  let stub: StructuralEngineStub;
  let liveEngine: Engine;
  const live = () => getDocumentLayer(stub.document(), 'control-1') as CanvasControlLayerContract;

  const LiveSettings = () => {
    const layer = useSyncExternalStore(stub.subscribe, live);
    return <ControlLayerSettings engine={liveEngine} layer={layer} />;
  };

  const renderLive = async (adapter: Partial<CanvasControlLayerContract['adapter']>) => {
    stub = createStructuralEngineStub({
      layers: [
        detectedLayer('depth', 'depth_anything_depth_estimation', {
          adapter: { ...controlLayer().adapter, ...adapter },
        }),
      ],
      schedulePreview: (flush) => {
        flush();
        return () => undefined;
      },
    });
    liveEngine = {
      ...stub.engine,
      document: { ...stub.engine.document, getDocument: () => stub.document() },
      exports: { exportRasterComposite: () => ({ status: 'empty' }), hasExportableLayerContent: () => true },
      interaction: { get: () => false, subscribe: () => () => undefined },
      lifecycle: { flushPendingUploads: () => Promise.resolve() },
      tools: { setTool: vi.fn() },
    } as unknown as Engine;
    attachOperations(liveEngine);
    applyThemeToRoot(DEFAULT_THEME_ID);
    host = document.createElement('div');
    host.style.width = '320px';
    document.body.append(host);
    root = createRoot(host);
    await settle(() =>
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <QueryClientProvider client={queryClient}>
              <LiveSettings />
            </QueryClientProvider>
          </ChakraProvider>
        </I18nextProvider>
      )
    );
  };

  const dragSlider = async (
    thumbName: string,
    fractions: number[],
    onMove: () => void = () => undefined,
    { release = true } = {}
  ) => {
    const control = page
      .getByRole('slider', { name: thumbName })
      .element()
      .closest<HTMLElement>('[data-part="root"]')!
      .querySelector<HTMLElement>('[data-part="control"]')!;
    const rect = control.getBoundingClientRect();
    const pointer = (target: EventTarget, type: string, fraction: number) =>
      settle(() =>
        target.dispatchEvent(
          new PointerEvent(type, {
            bubbles: true,
            button: 0,
            clientX: rect.left + rect.width * fraction,
            clientY: rect.top + rect.height / 2,
            pointerId: 1,
          })
        )
      );
    const [first, ...rest] = fractions;
    await pointer(control, 'pointerdown', first!);
    for (const fraction of rest) {
      await pointer(document, 'pointermove', fraction);
      onMove();
    }
    if (release) {
      await pointer(document, 'pointerup', fractions.at(-1)!);
    }
  };
  const thumbValue = (name: string) =>
    Number(page.getByRole('slider', { name }).element().getAttribute('aria-valuenow'));

  it('follows a weight drag step by step and keeps the released weight as one undoable step', async () => {
    await renderLive({ weight: 1 });
    const seen: number[] = [];
    await dragSlider('Weight', [0.5, 0.6, 0.7, 0.8], () => {
      seen.push(thumbValue('Weight'));
      expect(live().adapter.weight).toBe(thumbValue('Weight'));
    });

    expect(seen).toHaveLength(3);
    expect(seen[0]).toBeGreaterThan(1);
    expect(seen[1]).toBeGreaterThan(seen[0]!);
    expect(seen[2]).toBeGreaterThan(seen[1]!);
    const released = seen.at(-1)!;
    expect(live().adapter.weight).toBe(released);
    expect(thumbValue('Weight')).toBe(released);
    await expect.element(page.getByRole('spinbutton', { name: 'Weight' })).toHaveValue(released.toFixed(2));
    expect(stub.history.entries().past).toEqual(['Weight']);
    expect(stub.commits).toHaveLength(1);
    expect(stub.commits[0]!.inverse).toMatchObject({ config: { adapter: { weight: 1 } } });

    await settle(() => stub.engine.history.undo());
    expect(live().adapter.weight).toBe(1);
    expect(thumbValue('Weight')).toBe(1);
    await settle(() => stub.engine.history.redo());
    expect(live().adapter.weight).toBe(released);
  });
});

describe('ControlLayerSettings and the capability table', () => {
  it('says the layer is checked once model capabilities load', async () => {
    accountLifecycle.invalidate();
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    expect(
      page
        .getByRole('status')
        .elements()
        .map((element) => element.textContent)
    ).toContain('This layer is checked once model capabilities have loaded.');
  });
});

describe('ControlLayerSettings intent refusals', () => {
  it.each([
    ['blocked', 'The canvas is busy. Try again when the current action finishes.'],
    ['locked', 'Unlock the layer before using this action.'],
    ['no-detection-input', 'Replace the image to detect again'],
    ['over-budget', 'Layer action failed'],
  ] as const)('says why a %s edit did not land', async (result, title) => {
    const created = vi.spyOn(toaster, 'create');
    try {
      intentResult = result;
      await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
      await choose('Detection', 'Normal map');

      expect(intents).toEqual([{ detector: 'normal_map', kind: 'set-detector' }]);
      expect(created).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ title }));
    } finally {
      created.mockRestore();
    }
  });
});
