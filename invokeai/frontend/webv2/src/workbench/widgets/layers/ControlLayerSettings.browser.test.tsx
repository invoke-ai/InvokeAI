import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasLayerPreviewMutation,
} from '@workbench/canvas-engine/api';
import type { ControlLayerIntent, ControlLayerIntentResult } from '@workbench/canvas-operations/api';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { toaster } from '@platform/ui/toaster';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { createControlLayer, createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { previewInverse } from '@workbench/canvas-engine/document-model/documentModel';
import { createControlDetectionStatusStore } from '@workbench/canvas-operations/controlDetectionStatusStore';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { resetPropertyGroupCollapse } from '@workbench/widgets/canvas/tool-presentation/propertyGroupStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider } from 'react-i18next';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ControlLayerSettings } from './ControlLayerSettings';

const harness = vi.hoisted(() => ({
  canManageModels: true,
  mainModel: { base: 'sdxl', key: 'sdxl-main', name: 'SDXL', type: 'main' } as Record<string, unknown> | null,
  memory: [] as [string, string | null][],
  startersRequested: 0,
  models: [] as Record<string, unknown>[],
  navigate: [] as unknown[],
}));

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // The real one fetches the starter catalog.
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
  // The real one pushes preferences to the server.
  patchControlModelMemory: (key: string, model: string | null) => {
    harness.memory.push([key, model]);
    return Promise.resolve();
  },
}));
const getArchitectureCapabilities = vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>();
vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
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
const T2I_CANNY = model('T2I Canny XL', 't2i_adapter');

const PHOTO = { image: { height: 64, imageName: 'photo.png', width: 96 }, type: 'image' } as const;

const controlLayer = (overrides: Partial<CanvasControlLayerContract> = {}): CanvasControlLayerContract => {
  const created = createControlLayer({ base: 'sdxl', id: 'control-1', name: 'Control Layer 1' });
  return { ...created, ...overrides, adapter: { ...created.adapter, ...overrides.adapter } };
};
/** A typed layer detected from a photo, its map current. */
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
  const documentModel = createDocumentModel(document, { editRevision: 0, projectId: 'test-project' });
  const engine = {
    document: { getDocument: () => document, model: () => documentModel },
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
            return true;
          },
          baseline: () => baseline,
          cancel: () => {
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
  return engine;
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

const ROW_LABELS = ['Image', 'Type', 'Detection', 'Model', 'Weight', 'Steps', 'Priority'];
/** Where each row's label sits, relative to the panel. */
const rowTops = () => {
  const top = host!.getBoundingClientRect().top;
  return ROW_LABELS.map((label) => {
    const element = [...host!.querySelectorAll('p')].find((candidate) => candidate.textContent === label);
    if (!element) {
      throw new Error(`No row labelled ${label}`);
    }
    return Math.round(element.getBoundingClientRect().top - top);
  });
};
const combobox = (name: string) => page.getByRole('combobox', { exact: true, name });
const choose = async (name: string, option: string | RegExp) => {
  await act(() => combobox(name).click());
  await act(() => page.getByRole('option', { name: option }).click());
  await settle();
};
const adapterOf = (index: number) =>
  (commits[index]!.edit as { forward: { config: { adapter: Record<string, unknown> } } }).forward.config.adapter;

describe('ControlLayerSettings rows', () => {
  it('keeps every row where it is across the empty, ready, warning, choosing and detecting states', async () => {
    const states: [CanvasControlLayerContract, () => void][] = [
      [controlLayer({}), () => undefined],
      [detectedLayer('depth', 'depth_anything_depth_estimation'), () => undefined],
      [controlLayer({ source: PHOTO }), () => undefined],
      [
        detectedLayer('depth', 'depth_anything_depth_estimation'),
        () => {
          // Two depth models: the Model row asks, and its caption runs to two lines.
          harness.models = [DEPTH_A, DEPTH_B, POSE];
        },
      ],
      [
        detectedLayer('depth', 'depth_anything_depth_estimation'),
        () => statuses.set('control-1', { detector: 'depth_anything_depth_estimation', status: 'processing' }),
      ],
    ];
    // Captions wrap differently at each width; the rows must not care.
    for (const width of [320, 450]) {
      const tops: number[][] = [];
      harness.models = [DEPTH_A, POSE];
      statuses.clear();
      for (const [layer, prepare] of states) {
        prepare();
        await render(layer, { width });
        tops.push(rowTops());
        expect(host!.scrollWidth, 'no horizontal overflow').toBeLessThanOrEqual(host!.clientWidth);
        await unmount();
      }
      expect(
        tops.map((entry) => entry.join(',')),
        `at ${width} px`
      ).toEqual(tops.map(() => tops[0]!.join(',')));
      // The fixed rows fit the Properties pane budget.
      expect(tops[0]!.at(-1)!).toBeLessThan(281);
    }
  });

  it('passes an accessibility audit empty, with a warning and ready', async () => {
    for (const layer of [
      controlLayer({}),
      controlLayer({ source: PHOTO }),
      detectedLayer('depth', 'depth_anything_depth_estimation'),
    ]) {
      await render(layer);
      expect(await auditAccessibility(host!)).toEqual([]);
      await unmount();
    }
  });

  it('flags an image without a type and explains that the model follows the type', async () => {
    await render(controlLayer({ source: PHOTO }));

    const type = combobox('Type');
    await expect.element(type).toHaveAttribute('aria-invalid', 'true');
    await expect.element(type).toHaveAccessibleDescription('Choose a type and the model follows.');
    await expect.element(combobox('Detection')).toBeDisabled();

    // A missing type warns; it is not an error, so neither the border nor the focus ring turns red.
    const trigger = type.element() as HTMLElement;
    // The placeholder is written in the warning colour; the border matches it.
    const warning = getComputedStyle(trigger.querySelector('[data-part="value-text"]')!).color;
    expect(getComputedStyle(trigger).borderTopColor).toBe(warning);
    // Keyboard focus, so the focus-visible ring applies.
    trigger.focus();
    await act(() => userEvent.keyboard('{Shift>}{Tab}{/Shift}{Tab}'));
    expect(document.activeElement).toBe(trigger);
    expect(getComputedStyle(trigger).borderTopColor).toBe(warning);
  });
});

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

  it('keeps a name the user gave the layer', async () => {
    await render(controlLayer({ name: 'Background pose', source: PHOTO }));
    await choose('Type', /^Pose/);

    expect(intents).toEqual([{ controlType: 'pose', kind: 'set-type', name: undefined }]);
  });

  it("offers the type's detectors, then None after a divider", async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    await act(() => combobox('Detection').click());
    const options = [...document.querySelectorAll<HTMLElement>('[role="option"]')].map((item) => item.textContent);
    expect(options).toEqual(['Depth Anything', 'Normal map', 'None: image is already a depth map']);
    await act(() => page.getByRole('option', { name: 'Normal map' }).click());
    await settle();
    await choose('Detection', 'None: image is already a depth map');

    expect(intents).toEqual([
      { detector: 'normal_map', kind: 'set-detector' },
      { detector: null, kind: 'set-detector' },
    ]);
  });

  it('shows a spinner while detection runs, then offers Run again with the failure reason', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    await settle(() =>
      statuses.set('control-1', { detector: 'depth_anything_depth_estimation', status: 'processing' })
    );
    expect(combobox('Detection').element().querySelector('.chakra-spinner')).not.toBeNull();

    await settle(() => statuses.set('control-1', { error: { code: 'process-failed' }, status: 'error' }));
    expect(combobox('Detection').element().querySelector('.chakra-spinner')).toBeNull();
    // The canvas announces runs; the row keeps the failure and its reason on the retry.
    await userEvent.hover(page.getByRole('button', { name: 'Run again' }));
    await expect.element(page.getByText('Depth Anything failed: the detector could not run')).toBeVisible();
    await act(() => page.getByRole('button', { name: 'Run again' }).click());

    expect(retries).toEqual(['control-1']);
  });

  it('offers Detect again for an out-of-date map once nothing is running', async () => {
    // The map was detected with other settings, as after an undo of a settings change.
    await render(
      detectedLayer('depth', 'depth_anything_depth_estimation', {
        filter: { settings: { model_size: 'large' }, type: 'depth_anything_depth_estimation' },
      })
    );
    await act(() => page.getByRole('button', { name: 'Detect again' }).click());

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

  it('says why None has no settings', async () => {
    await render(
      detectedLayer('tile', 'canny_edge_detection', { detection: null, filter: undefined, original: undefined })
    );

    const toggle = page.getByRole('button', { name: 'Detector settings' });
    await expect.element(toggle).toHaveAttribute('aria-disabled', 'true');
    await expect.element(toggle).toHaveAccessibleDescription('No settings for None: Tile uses the image directly');
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

  it('asks for the starter catalog when the list opens, and leaves origins out until it arrives', async () => {
    harness.models = [DEPTH_A, DEPTH_B];
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
    expect(harness.startersRequested).toBe(0);

    await act(() => combobox('Model').click());

    expect(harness.startersRequested).toBe(1);
    await expect
      .element(page.getByRole('option', { exact: true, name: 'controlnet-zoe-depth-sdxl, ControlNet' }))
      .toBeVisible();
  });

  it('describes the Model choice with its caption', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    await expect.element(combobox('Model')).toHaveAccessibleDescription('ControlNet · your only depth model');
  });

  it('resets a pick back to automatic and returns focus to the model choice', async () => {
    harness.models = [DEPTH_A, DEPTH_B];
    const picked = detectedLayer('depth', 'depth_anything_depth_estimation', {
      adapter: { ...controlLayer().adapter, model: 'controlnet-zoe-depth-sdxl' },
    });
    await render(picked);
    expect(host!.textContent).toContain('ControlNet · your pick for depth');

    await act(() => page.getByRole('button', { name: 'Reset' }).click());
    await settle();

    expect(adapterOf(0)).toEqual({ model: null });
    expect(harness.memory).toEqual([['sdxl:depth', null]]);
    expect(document.activeElement).toBe(combobox('Model').element());
  });

  it('names the only model, and points to the Model Manager when none fits', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
    expect(host!.textContent).toContain('ControlNet · your only depth model');
    await unmount();

    await render(detectedLayer('lineart', 'lineart_edge_detection'));
    expect(combobox('Model').element().textContent).toContain('None for SDXL');
    await act(() => page.getByRole('button', { name: 'Find one' }).click());
    expect(harness.navigate).toEqual([{ search: { project: 'test-project' }, to: '/models' }]);
    await unmount();

    // Without the right to manage models, the caption says it without the link.
    harness.canManageModels = false;
    harness.models = [];
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
    expect(host!.textContent).toContain('No control models are installed for SDXL.');
    expect(page.getByRole('button', { name: 'Install starter models' }).elements()).toHaveLength(0);
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
    // The release lands within a step of where the drag ended, recorded from where it began.
    expect(forward.config.adapter.weight).toBeCloseTo(1.5, 0);
    expect(forward.config.adapter.weight).toBe(
      (previews.at(-1) as { config: { adapter: { weight: number } } }).config.adapter.weight
    );
    expect(inverse.config.adapter.weight).toBe(1);
  });

  it('clamps typed weights to the bounds the model allows', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
    const input = page.getByRole('spinbutton', { name: 'Weight' });
    await act(async () => {
      await userEvent.clear(input);
      await userEvent.type(input, '-3{Enter}');
    });
    await act(async () => {
      await userEvent.clear(input);
      await userEvent.type(input, '9{Enter}');
    });

    expect(commits.map((_, index) => adapterOf(index).weight)).toEqual([-1, 2]);
  });

  it('reads out the step range', async () => {
    await render(
      detectedLayer('depth', 'depth_anything_depth_estimation', {
        adapter: { ...controlLayer().adapter, beginEndStepPct: [0.1, 0.75] },
      })
    );

    expect(host!.textContent).toContain('10–75%');
  });

  it('sets the priority for a ControlNet and dims it, with the reason, for a T2I Adapter', async () => {
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));
    await act(() => page.getByRole('radiogroup', { name: 'Priority' }).getByText('Control', { exact: true }).click());
    await settle();
    expect(adapterOf(0)).toEqual({ controlMode: 'more_control' });
    await unmount();

    harness.models = [T2I_CANNY];
    await render(detectedLayer('edges', 'canny_edge_detection'));
    const group = page.getByRole('radiogroup', { name: 'Priority' });
    await expect.element(group).toHaveAccessibleDescription('Priority applies to ControlNet models only.');
    await expect.element(page.getByRole('radio', { name: 'Control' })).toBeDisabled();
  });
});

describe('ControlLayerSettings and the capability table', () => {
  const retryButton = () => [...host!.querySelectorAll('button')].find((button) => button.textContent === 'Retry');

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

  it('keeps the retry and its focus through a retry that fails again, then hands focus to the panel', async () => {
    accountLifecycle.invalidate();
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render(detectedLayer('depth', 'depth_anything_depth_estimation'));

    let failLoad: (error: Error) => void = () => undefined;
    getArchitectureCapabilities.mockReturnValueOnce(
      new Promise((_, reject) => {
        failLoad = reject;
      })
    );
    await settle(() => {
      retryButton()?.focus();
      retryButton()?.click();
    });
    expect(document.activeElement).toBe(retryButton());
    expect(retryButton()?.getAttribute('aria-busy')).toBe('true');

    await settle(() => failLoad(new Error('Still out.')));
    expect(document.activeElement).toBe(retryButton());

    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(() => retryButton()?.click());

    expect(retryButton()).toBeUndefined();
    expect(host!.contains(document.activeElement)).toBe(true);
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
