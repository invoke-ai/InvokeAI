import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type { CanvasControlLayerContract, CanvasLayerPreviewMutation } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { previewInverse } from '@workbench/canvas-engine/document-model/documentModel';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ControlLayerSettings } from './ControlLayerSettings';
import { createControlLayer } from './layerOps';

// Stable identities, as the real stores hand out, so only the table's arrival can re-read the policy.
const catalog = vi.hoisted(() => ({
  mainModel: { base: 'sd-1', key: 'sd1-main', name: 'SD 1.5', type: 'main' } as Record<string, unknown>,
  models: [{ base: 'sd-1', key: 'sd1-controlnet', name: 'SD 1.5 ControlNet', type: 'controlnet' }] as Record<
    string,
    unknown
  >[],
}));
// Tests that swap the catalog restore these identities afterwards.
const SD1_CATALOG = { ...catalog };

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useModelsSelector: (selector: (snapshot: { models: typeof catalog.models }) => unknown) =>
    selector({ models: catalog.models }),
}));
vi.mock('./useSelectedMainModel', () => ({ useSelectedMainModel: () => catalog.mainModel }));

const getArchitectureCapabilities = vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>();
vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const layer = {
  adapter: {
    beginEndStepPct: [0, 1],
    controlMode: 'balanced',
    kind: 'controlnet',
    model: 'sd1-controlnet',
    weight: 1,
  },
  filter: null,
  id: 'control-1',
  isEnabled: true,
  isLocked: false,
  name: 'Control 1',
  opacity: 1,
  type: 'control',
  withTransparencyEffect: false,
} as unknown as CanvasControlLayerContract;

const ignoreOperationStarted = (): void => undefined;

const settle = async (run: () => void = () => undefined) => {
  await act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

type Engine = NonNullable<Parameters<typeof ControlLayerSettings>[0]['engine']>;

/** Committed document edits, in order; each carries the adapter patch the panel prepared. */
const commits: { label: string; edit: unknown }[] = [];

/** Live previews the panel sent, and the baselines a cancelled gesture restored. */
const previews: CanvasLayerPreviewMutation[] = [];
const restores: CanvasLayerPreviewMutation[] = [];
let editingLocked = false;
let previewsRefused = false;

/** Just enough engine for the panel's reads and commits: every layer has content and contributes, in order. */
const engineWith = (layers: CanvasControlLayerContract[]): Engine => {
  const model = createDocumentModel(
    { ...createEmptyCanvasDocument(), stacks: stacksFrom(layers) },
    { editRevision: 0, projectId: 'test-project' }
  );
  const engine = {
    document: { model: () => model },
    exports: { hasExportableLayerContent: () => true },
    interaction: { get: () => editingLocked, subscribe: () => () => undefined },
    layers: {
      beginStructuralPreview: () => {
        if (previewsRefused) {
          return null;
        }
        // As the engine's session: the first preview captures what it replaces, and cancel restores it.
        let baseline: CanvasLayerPreviewMutation | null = null;
        return {
          apply: (action: CanvasLayerPreviewMutation) => {
            baseline ??= previewInverse(model.getNode(action.id)!, action);
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
  } as unknown as Engine;
  attachCanvasOperations(engine, {} as never);
  return engine;
};

const render = async (controlLayer: CanvasControlLayerContract = layer, engine: Engine | null = null) => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <ControlLayerSettings engine={engine} layer={controlLayer} onOperationStarted={ignoreOperationStarted} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

/** The kind Select renders a hidden native select, so its offered kinds are readable without opening it. */
const offersKind = (kind: string) => host!.querySelector(`option[value="${kind}"]`) !== null;
const textOf = (role: 'alert' | 'status') =>
  [...host!.querySelectorAll(`[role="${role}"]`)].map((element) => element.textContent).join(' ');
const retryButton = () => [...host!.querySelectorAll('button')].find((button) => button.textContent === 'common.retry');

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  getArchitectureCapabilities.mockReset();
  Object.assign(catalog, SD1_CATALOG);
  commits.length = 0;
  previews.length = 0;
  restores.length = 0;
  editingLocked = false;
  previewsRefused = false;
  // Returns the capability store and the core registry to their unloaded state between tests.
  accountLifecycle.invalidate();
});

describe('ControlLayerSettings and the capability table', () => {
  it('says the adapter is waiting for model capabilities rather than that it is unsupported', async () => {
    await render();

    expect(textOf('status')).toContain('widgets.layers.control.capabilitiesLoading');
  });

  it('keeps the retry and its focus through a retry that fails again, then offers the adapter kinds', async () => {
    // Capability arrival must update kinds even when the model base is unchanged.
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render();

    expect(textOf('alert')).toContain('widgets.layers.control.capabilitiesLoadFailed');
    expect(offersKind('controlnet')).toBe(false);

    // A retry that fails again: the button never unmounts, so a keyboard user's focus stays on it.
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

    expect(offersKind('controlnet')).toBe(true);
    expect(retryButton()).toBeUndefined();
    // The button that held focus is gone; focus moved into the panel instead of falling to <body>.
    expect(document.activeElement).not.toBe(document.body);
    expect(host!.contains(document.activeElement)).toBe(true);
  });

  it('leaves focus where the user moved it while the retry was in flight', async () => {
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render();

    let finishLoad: (rows: ArchitectureCapabilitiesRow[]) => void = () => undefined;
    getArchitectureCapabilities.mockReturnValueOnce(
      new Promise((resolve) => {
        finishLoad = resolve;
      })
    );
    await settle(() => {
      retryButton()?.focus();
      retryButton()?.click();
    });

    // Someone starts typing elsewhere; the load landing must not pull them into this panel.
    const elsewhere = document.createElement('input');
    document.body.append(elsewhere);
    elsewhere.focus();
    try {
      await settle(() => finishLoad(architectureCapabilitiesFixture));

      expect(retryButton()).toBeUndefined();
      expect(document.activeElement).toBe(elsewhere);
    } finally {
      elsewhere.remove();
    }
  });
});

describe('ControlLayerSettings on an Anima main model', () => {
  const lllite = (key: string, name: string, cond_in_channels: number | null) => ({
    base: 'anima',
    cond_in_channels,
    key,
    name,
    type: 'controlnet',
  });
  const ANIMA_MODELS = [
    lllite('sketch', 'Anima LLLite Sketch', 3),
    lllite('inpainting', 'Anima LLLite Inpainting', 4),
    lllite('unidentified', 'Anima LLLite (old install)', null),
  ];
  const animaLayer = (
    kind: string,
    model: string | null,
    id = 'control-1',
    values: Partial<CanvasControlLayerContract['adapter']> = {}
  ): CanvasControlLayerContract => {
    const base = createControlLayer(id, id, 'anima', model);
    return {
      ...base,
      adapter: { ...base.adapter, ...values, kind: kind as CanvasControlLayerContract['adapter']['kind'] },
    };
  };
  /** The model Select's hidden native options, as the kind ones are read. */
  const offeredModels = () =>
    [...host!.querySelectorAll('select')]
      .flatMap((select) => [...select.options].map((option) => option.value))
      .filter((value) => catalog.models.some((model) => model.key === value));
  const button = (name: string) => [...host!.querySelectorAll('button')].find((b) => b.textContent === name);
  /** The adapter each commit wrote, read from the prepared edit's forward patch. */
  const committedAdapters = () =>
    commits
      .map(({ edit }) => JSON.stringify(edit))
      .map((json) => {
        const match = /"adapter":(\{[^{}]*"beginEndStepPct":\[[^\]]*\][^{}]*\})/u.exec(json);
        return match ? (JSON.parse(match[1]!) as Record<string, unknown>) : null;
      });

  /** Renders `layers.at(-1)` as the selected layer of a document holding all of them, each with content. */
  const renderWith = async (
    mainModel: Record<string, unknown> | null,
    models: Record<string, unknown>[],
    ...layers: CanvasControlLayerContract[]
  ) => {
    catalog.mainModel = mainModel as Record<string, unknown>;
    catalog.models = models;
    commits.length = 0;
    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(ensureArchitectureCapabilitiesLoaded);
    await render(layers.at(-1), engineWith(layers));
  };
  const ANIMA_MAIN = { base: 'anima', key: 'anima-main', name: 'Anima', type: 'main' };
  const renderAnima = (...layers: CanvasControlLayerContract[]) => renderWith(ANIMA_MAIN, ANIMA_MODELS, ...layers);

  it('offers ControlNet-LLLite and its control adapters, and accepts one without a warning', async () => {
    await renderAnima(animaLayer('anima_lllite', 'sketch'));

    expect(offersKind('anima_lllite')).toBe(true);
    expect(offersKind('controlnet')).toBe(false);
    expect(offeredModels()).toEqual(['sketch']);
    expect(host!.textContent).toContain('Anima LLLite Sketch');
    expect(textOf('alert')).toBe('');
  });

  it('turns a layer saved as ControlNet into LLLite in one step, keeping its model, weight and steps', async () => {
    await renderAnima(animaLayer('controlnet', 'sketch', 'old', { beginEndStepPct: [0.1, 0.7], weight: 0.6 }));

    // The alert names the fix, and the model list does not present any model as usable under the wrong kind.
    expect(textOf('alert')).toContain('widgets.layers.control.validation.switch_adapter_kind');
    expect(offeredModels()).toEqual([]);
    expect(host!.textContent).toContain('Anima LLLite Sketch');

    await settle(() => button('widgets.layers.control.switchKind')?.click());

    expect(committedAdapters()).toEqual([
      { beginEndStepPct: [0.1, 0.7], controlMode: null, kind: 'anima_lllite', model: 'sketch', weight: 0.6 },
    ]);
  });

  it('asks for the switch, not a model, on a model-less layer saved as ControlNet', async () => {
    await renderAnima(animaLayer('controlnet', null));

    expect(textOf('alert')).toContain('widgets.layers.control.validation.switch_adapter_kind');
    expect(textOf('alert')).not.toContain('missing_model');
  });

  it('switches through the Adapter type select the same way', async () => {
    await renderAnima(animaLayer('controlnet', 'sketch', 'old', { beginEndStepPct: [0.1, 0.7], weight: 0.6 }));

    await act(() => page.getByRole('combobox', { name: 'widgets.layers.control.kind' }).click());
    await act(() => page.getByRole('option', { name: 'widgets.layers.control.kinds.anima_lllite' }).click());
    await settle();

    expect(committedAdapters()).toEqual([
      { beginEndStepPct: [0.1, 0.7], controlMode: null, kind: 'anima_lllite', model: 'sketch', weight: 0.6 },
    ]);
  });

  it('explains an empty LLLite list instead of only asking for a model', async () => {
    await renderWith(ANIMA_MAIN, ANIMA_MODELS.slice(1), animaLayer('anima_lllite', null));

    expect(offeredModels()).toEqual([]);
    expect(host!.textContent).toContain('widgets.layers.control.hiddenModels.lllite_inpaint_adapter');
    expect(host!.textContent).toContain('widgets.layers.control.hiddenModels.lllite_channels_unknown');
    expect(textOf('alert')).toBe('');
  });

  it('flags a second layer applying the same LLLite model, but not the first', async () => {
    const first = animaLayer('anima_lllite', 'sketch', 'first');
    const second = animaLayer('anima_lllite', 'sketch', 'second');
    await renderAnima(first, second);
    expect(textOf('alert')).toBe('widgets.layers.control.validation.duplicate_lllite_model');

    await act(() => root?.unmount());
    host?.remove();
    await render(first, engineWith([first, second]));
    expect(textOf('alert')).toBe('');
  });

  it('names an unusable adapter by its catalog name beside the alert about it', async () => {
    await renderAnima(animaLayer('anima_lllite', 'inpainting'));

    expect(textOf('alert')).toBe('widgets.layers.control.validation.lllite_inpaint_adapter');
    expect(host!.textContent).toContain('Anima LLLite Inpainting');
    expect(host!.textContent).not.toContain('widgets.layers.control.selectModel');
  });

  it('offers ControlNet for an LLLite layer once the main model is not Anima, dropping the LLLite model', async () => {
    await renderWith(
      { base: 'sdxl', key: 'sdxl-main', name: 'SDXL', type: 'main' },
      [...ANIMA_MODELS, { base: 'sdxl', key: 'sdxl-control', name: 'SDXL ControlNet', type: 'controlnet' }],
      animaLayer('anima_lllite', 'sketch', 'lllite', { beginEndStepPct: [0.2, 0.8], weight: 0.9 })
    );
    expect(textOf('alert')).toContain('widgets.layers.control.validation.switch_adapter_kind');

    await settle(() => button('widgets.layers.control.switchKind')?.click());

    expect(committedAdapters()).toEqual([
      { beginEndStepPct: [0.2, 0.8], controlMode: 'balanced', kind: 'controlnet', model: null, weight: 0.9 },
    ]);
  });

  it('hides the base-bound LLLite kind while no main model is selected', async () => {
    await renderWith(null, ANIMA_MODELS, animaLayer('controlnet', null));

    expect(offersKind('controlnet')).toBe(true);
    expect(offersKind('anima_lllite')).toBe(false);
  });
});

describe('ControlLayerSettings weight', () => {
  // A full document layer (the panel reads its leaf), weighted 1 so a drag can move either way.
  const weighted = ((): CanvasControlLayerContract => {
    const base = createControlLayer('Control 1', 'control-1', 'sd-1', 'sd1-controlnet');
    return { ...base, adapter: { ...base.adapter, weight: 1 } };
  })();
  const weightSlider = () => page.getByRole('slider', { name: 'widgets.layers.control.weight' });
  const weightOf = (edit: unknown, side: 'forward' | 'inverse') =>
    (edit as Record<typeof side, { config: { adapter: { weight: number } } }>)[side].config.adapter.weight;
  /** One drag on the weight scrubber through each offset (fractions of its 0–2 track), released at the last. */
  const dragWeight = async (offsets: number[]) => {
    const frame = weightSlider().element().closest<HTMLElement>('[data-scope="scrubber"]')!;
    const rect = frame.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const xAt = (offset: number) => startX + offset * (rect.width - 20);
    const pointer = (target: EventTarget, type: string, x: number) =>
      settle(() =>
        target.dispatchEvent(new PointerEvent(type, { bubbles: true, button: 0, clientX: x, pointerId: 1 }))
      );
    await pointer(frame, 'pointerdown', startX);
    for (const offset of offsets) {
      await pointer(window, 'pointermove', xAt(offset));
    }
    await pointer(window, 'pointerup', xAt(offsets.at(-1) ?? 0));
  };
  const typeWeight = async (text: string) => {
    await act(async () => {
      (weightSlider().element() as HTMLElement).focus();
      await userEvent.keyboard(`{Enter}${text}{Enter}`);
    });
  };

  it('previews a drag and records it as one step from the weight it started at', async () => {
    await render(weighted, engineWith([weighted]));
    await dragWeight([0.1, 0.25]);

    expect(previews.length).toBeGreaterThanOrEqual(2);
    expect(commits).toHaveLength(1);
    expect(weightOf(commits[0]!.edit, 'forward')).toBe(1.5);
    expect(weightOf(commits[0]!.edit, 'inverse')).toBe(1);
  });

  it('records nothing and restores the weight when a drag returns to where it started', async () => {
    await render(weighted, engineWith([weighted]));
    await dragWeight([0.25, 0]);

    expect(commits).toHaveLength(0);
    expect(restores).toEqual([
      { config: { adapter: { weight: 1 }, layerType: 'control' }, id: 'control-1', type: 'updateCanvasLayerConfig' },
    ]);
  });

  it('clamps typed weights to the typed bounds, which reach past the track', async () => {
    await render(weighted, engineWith([weighted]));
    await typeWeight('-3');

    expect(commits.map(({ edit }) => weightOf(edit, 'forward'))).toEqual([-1]);

    await typeWeight('9');

    expect(commits.map(({ edit }) => weightOf(edit, 'forward'))).toEqual([-1, 2]);
  });

  it('still attempts a typed weight when previews are refused, so the refusal is reported', async () => {
    previewsRefused = true;
    await render(weighted, engineWith([weighted]));
    await typeWeight('0.5');

    expect(commits.map(({ edit }) => weightOf(edit, 'forward'))).toEqual([0.5]);
  });

  it('disables the weight while document editing is locked', async () => {
    editingLocked = true;
    await render(weighted, engineWith([weighted]));

    await expect.element(weightSlider()).toHaveAttribute('aria-disabled', 'true');
  });
});
