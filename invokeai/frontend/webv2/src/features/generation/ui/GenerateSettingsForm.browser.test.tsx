/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
import type { GenerateLora, GenerateModelConfig, LoraModelConfig } from '@features/generation/core/types';
import type { ComponentType } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
import { dynamicPromptsKeys } from '@features/generation/data/dynamicPromptsQueries';
import { promptTemplatesQueryOptions } from '@features/generation/data/promptTemplates';
import { wildcardsQueryOptions } from '@features/generation/data/wildcards';
import { flushGenerateDrafts } from '@features/generation/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createExternalStoreCore, type ExternalStoreCore } from '@platform/state/externalStoreCore';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, Fragment } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GenerateWidgetView } from './GenerateWidgetView';
import { GenerationUiProvider, type GenerationModelSelectProps, type GenerationUiAdapter } from './GenerationUiContext';
import { requestGenerateModelSwitchConfirmation } from './modelSelectionNotice';

vi.mock('react-i18next', () => {
  // Shows the cleared-settings list a model-switch confirmation formats into its body.
  const t = (key: string, options?: { labels?: string }) => (options?.labels ? `${key}: ${options.labels}` : key);
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});

vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => Promise.resolve(architectureCapabilitiesFixture),
}));

/** Counts commits in which anything inside a real section rendered. */
const { commits, profiled } = vi.hoisted(() => {
  const commits = new Map<string, number>();
  const profiled =
    <Props extends object>(id: string, Component: ComponentType<Props>) =>
    async () => {
      const { createElement, Profiler } = await import('react');
      const onRender = () => commits.set(id, (commits.get(id) ?? 0) + 1);
      return (props: Props) => createElement(Profiler, { id, onRender }, createElement(Component, props));
    };
  return { commits, profiled };
});

type SectionModule = Record<string, ComponentType<object>>;

vi.mock('./GenerateModelCard', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { GenerateModelCard: await profiled('model', module.GenerateModelCard)() };
});
vi.mock('./promptFields', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { ...module, GeneratePromptFields: await profiled('prompts', module.GeneratePromptFields)() };
});
vi.mock('./GenerateDimensionFields', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { ...module, GenerateDimensionFields: await profiled('dimensions', module.GenerateDimensionFields)() };
});
vi.mock('./GenerateGuidanceSection', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { GenerateGuidanceSection: await profiled('guidance', module.GenerateGuidanceSection)() };
});
vi.mock('./GenerateRenderSection', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { GenerateRenderSection: await profiled('render', module.GenerateRenderSection)() };
});
vi.mock('./GenerateComponentsSection', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { GenerateComponentsSection: await profiled('components', module.GenerateComponentsSection)() };
});
vi.mock('./GenerateAdvancedFields', async (importOriginal) => {
  const module = await importOriginal<SectionModule>();
  return { GenerateAdvancedFields: await profiled('advanced', module.GenerateAdvancedFields)() };
});

const MODEL = {
  base: 'sdxl',
  format: 'diffusers',
  key: 'sdxl',
  name: 'SDXL',
  type: 'main',
} as GenerationModelCatalogItem;
const ink: LoraModelConfig = { base: 'sdxl', key: 'ink', name: 'Ink', type: 'lora' };
const chalk: LoraModelConfig = { base: 'sdxl', key: 'chalk', name: 'Chalk', type: 'lora' };
// The concepts the tests apply are installed, so their rows run.
const CATALOG: readonly GenerationModelCatalogItem[] = [MODEL, ink, chalk] as unknown as GenerationModelCatalogItem[];
const OTHER_SDXL = { ...MODEL, key: 'sdxl-2', name: 'SDXL 2' } as GenerationModelCatalogItem;
// A coarser size grid than SDXL's, and no SDXL LoRAs.
const FLUX = { ...MODEL, base: 'flux', key: 'flux', name: 'FLUX' } as GenerationModelCatalogItem;
const ANIMA = { ...MODEL, base: 'anima', key: 'anima', name: 'Anima' } as GenerationModelCatalogItem;
const SDXL_VAE = { base: 'sdxl', key: 'sdxl-vae', name: 'SDXL VAE', type: 'vae' } as GenerationModelCatalogItem;
const ANIMA_INPAINT_ADAPTER = {
  base: 'anima',
  cond_in_channels: 4,
  key: 'anima-inpaint',
  name: 'Anima LLLite Inpainting',
  type: 'controlnet',
} as GenerationModelCatalogItem;
const ANIMA_SKETCH_ADAPTER = {
  ...ANIMA_INPAINT_ADAPTER,
  cond_in_channels: 3,
  key: 'anima-sketch',
  name: 'Anima LLLite Sketch',
} as GenerationModelCatalogItem;

/**
 * The picker, reduced to buttons that hand a model to its real selection handler: the model card's main picks, and
 * for a component slot the catalog models of its types that pass its filter.
 */
const MainModelPicker = ({
  filter,
  id,
  invalid,
  modelTypes,
  onChange,
  placeholder,
  value,
}: GenerationModelSelectProps) =>
  modelTypes.includes('external_image_generator') ? (
    <>
      {/* The trigger the card's "Choose another model" opens. */}
      <button id={id} data-main-picker-trigger="" type="button" onClick={mainPickerOpened}>
        {placeholder}
      </button>
      {[OTHER_SDXL, FLUX].map((model) => (
        <button key={model.key} data-model-pick={model.key} type="button" onClick={() => onChange(model)}>
          {model.name}
        </button>
      ))}
    </>
  ) : (
    <div
      aria-invalid={invalid}
      data-component-placeholder={placeholder}
      data-component-types={modelTypes.join(',')}
      data-component-value={value ?? ''}
    >
      {stableGroups.models.catalog
        .filter((model) => modelTypes.includes(model.type) && (filter?.(model) ?? true))
        .map((model) => (
          <button key={model.key} data-component-pick={model.key} type="button" onClick={() => onChange(model)}>
            {model.name}
          </button>
        ))}
    </div>
  );
const INITIAL_VALUES: Record<string, unknown> = {
  ...getDefaultGenerateSettings(MODEL as never),
  modelKey: MODEL.key,
  positivePrompt: 'a lighthouse',
  steps: 30,
};
const UNRELATED_SECTIONS = ['model', 'prompts', 'dimensions', 'guidance', 'components', 'advanced'];

const noop = () => undefined;
const mainPickerOpened = vi.fn();
const ALL_SECTIONS_OPEN = new Proxy({}, { get: () => true }) as Record<string, boolean>;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient = new QueryClient();
let projectValues = new Map<string, ExternalStoreCore<Record<string, unknown>>>();
let patches: Array<{ patch: Record<string, unknown>; projectId: string | undefined }> = [];
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const storedValues = (projectId = 'project-1') => {
  const values = projectValues.get(projectId);

  if (!values) {
    throw new Error(`no stored values for ${projectId}`);
  }

  return values;
};

const NO_QUEUE_INSIGHTS = { secondsPerRun: null, seedHistory: [] };

/** Groups shared by every adapter a test renders, as the app keeps them stable across renders. */
const createStableGroups = () => ({
  CanvasDenoisingStrength: () => null,
  CanvasGenerationSections: () => null,
  CanvasRenderSize: () => null,
  account: { currentUserId: null, multiuserEnabled: false },
  aspectRatioPresets: { isFull: false, presets: [], remove: noop, save: () => null, update: noop },
  capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
  gallery: { findImage: noop, getUploadBoardId: () => Promise.resolve('none'), selectedImage: null, touchImages: noop },
  models: {
    ModelSelect: MainModelPicker,
    catalog: CATALOG,
    ensureLoaded: noop,
    error: null as string | null,
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
    openManager: noop,
    refresh: () => Promise.resolve(),
    status: 'loaded',
  },
  notifications: { error: noop, info: noop, reportError: noop, undoable: noop as (...args: unknown[]) => void },
  presets: { includePrompts: true, presets: [], remove: noop, rename: noop, save: noop, setIncludePrompts: noop },
  promptHistory: {
    clear: noop,
    items: [] as { negativePrompt: string | null; positivePrompt: string }[],
    remove: noop,
    restore: noop,
  },
  queueInsights: { getSnapshot: () => NO_QUEUE_INSIGHTS, subscribe: () => noop },
  rebalancePresets: { presets: [], remove: noop, rename: noop, save: noop },
  sectionPreferences: { sectionsOpen: ALL_SECTIONS_OPEN, setSectionOpen: noop },
  settings: {
    readGenerateValues: (projectId: string) => projectValues.get(projectId)?.getSnapshot() ?? null,
    patchGenerateSettings: (patch: Record<string, unknown>, projectId?: string) => {
      patches.push({ patch, projectId });
      storedValues(projectId).patchSnapshot(patch);
    },
  },
});
let stableGroups = createStableGroups();

/** The app adapter's shape: stable groups, with the active project's stored values behind a store port. */
const buildAdapter = (activeProjectId = 'project-1'): GenerationUiAdapter =>
  ({
    ...stableGroups,
    generateValues: storedValues(activeProjectId),
    project: { activeProjectId, invocationSourceId: 'generate', showPromptSyntaxHighlighting: false },
  }) as unknown as GenerationUiAdapter;

const settle = (run: () => void = noop, ms = 0) =>
  act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, ms);
    });
  });

const stepsScrubber = (): HTMLElement => {
  const label = [...document.querySelectorAll('[data-scope="scrubber"] [data-part="label"]')].find(
    (candidate) => candidate.textContent === 'widgets.generate.steps'
  );
  const scrubber = label?.closest<HTMLElement>('[data-scope="scrubber"]');

  if (!scrubber) {
    throw new Error('the Steps scrubber did not render');
  }

  return scrubber;
};

const stepsValue = () => Number(stepsScrubber().querySelector('[role="slider"]')?.getAttribute('aria-valuenow'));

const sliderNamed = (label: string): HTMLElement => {
  const scrubber = [...document.querySelectorAll('[data-scope="scrubber"]')].find(
    (candidate) => candidate.querySelector('[data-part="label"]')?.textContent === label
  );
  const slider = scrubber?.querySelector<HTMLElement>('[role="slider"]');

  if (!slider) {
    throw new Error(`the ${label} scrubber did not render`);
  }

  return slider;
};

const pressRight = (element: Element | null | undefined) =>
  settle(() => element?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));

const showConcepts = () =>
  settle(() =>
    storedValues().patchSnapshot({ loras: [ink, chalk].map((model) => ({ isEnabled: true, model, weight: 0.75 })) })
  );

const conceptRow = (name: string) => host!.querySelector<HTMLElement>(`[role="group"][aria-label="${name}"]`);

const committedSections = () => [...commits.keys()].sort();

/** `accountKey` stands in for the App remounting the authenticated tree when the account changes. */
const renderAdapter = (adapter: GenerationUiAdapter, accountKey = 'account-1') =>
  settle(() =>
    root?.render(
      <Fragment key={accountKey}>
        <QueryClientProvider client={queryClient}>
          <ChakraProvider value={system}>
            <DndContext>
              <GenerationUiProvider adapter={adapter}>
                <GenerateWidgetView />
              </GenerationUiProvider>
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </Fragment>
    )
  );

/** A keyboard step on the Steps slider: a debounced edit, like one scrub tick. */
const stepSteps = () =>
  settle(() =>
    stepsScrubber()
      .querySelector('[role="slider"]')
      ?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }))
  );

beforeEach(async () => {
  projectValues = new Map([
    ['project-1', createExternalStoreCore<Record<string, unknown>>(INITIAL_VALUES)],
    ['project-2', createExternalStoreCore<Record<string, unknown>>({ ...INITIAL_VALUES, steps: 50 })],
  ]);
  patches = [];
  stableGroups = createStableGroups();
  // Seeded, never-stale reads keep late responses from landing inside a counted window.
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  queryClient.setQueryData(wildcardsQueryOptions().queryKey, []);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderAdapter(buildAdapter());
  await settle(ensureArchitectureCapabilitiesLoaded, 50);
  commits.clear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  commits.clear();
  accountLifecycle.invalidate();
});

describe('GenerateSettingsForm Anima inpaint adapter', () => {
  const adapterSlot = () => host!.querySelector<HTMLElement>('[data-component-types="controlnet"]');
  const adapterPicks = () =>
    [...(adapterSlot()?.querySelectorAll('[data-component-pick]') ?? [])].map((pick) =>
      pick.getAttribute('data-component-pick')
    );
  const weightLabel = 'widgets.generate.animaInpaintAdapterWeight';
  const weightScrubber = () =>
    [...document.querySelectorAll('[data-scope="scrubber"]')].find(
      (candidate) => candidate.querySelector('[data-part="label"]')?.textContent === weightLabel
    );

  it('offers the slot to Anima only, lists only 4-channel adapters, and shows the weight once one is picked', async () => {
    stableGroups.models.catalog = [MODEL, ANIMA, ANIMA_INPAINT_ADAPTER, ANIMA_SKETCH_ADAPTER];
    await renderAdapter(buildAdapter());

    expect(adapterSlot()).toBeNull();

    await settle(() =>
      storedValues().patchSnapshot({ ...getDefaultGenerateSettings(ANIMA as never), modelKey: ANIMA.key })
    );

    expect(adapterPicks()).toEqual([ANIMA_INPAINT_ADAPTER.key]);
    expect(weightScrubber()).toBeUndefined();

    await settle(() => host!.querySelector<HTMLElement>('[data-component-pick="anima-inpaint"]')?.click());
    await settle(flushGenerateDrafts);

    expect(storedValues().getSnapshot()).toMatchObject({ animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER });
    const slider = sliderNamed(weightLabel);
    expect(slider.getAttribute('aria-valuenow')).toBe('1');
    expect(document.getElementById(slider.getAttribute('aria-labelledby') ?? '')?.textContent).toBe(weightLabel);

    await pressRight(sliderNamed(weightLabel));
    await settle(flushGenerateDrafts);

    expect(storedValues().getSnapshot()).toMatchObject({ animaInpaintAdapterWeight: 1.05 });
  });

  it('switches the adapter off and back on without losing the pick or the weight', async () => {
    const switchLabel = 'widgets.generate.animaInpaintAdapterEnabled';
    const offHelp = 'widgets.generate.animaInpaintAdapterOffHelp';
    const adapterSwitch = () =>
      [...host!.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find(
        (input) => input.labels?.[0]?.textContent === switchLabel
      );
    stableGroups.models.catalog = [MODEL, ANIMA, ANIMA_INPAINT_ADAPTER];
    await renderAdapter(buildAdapter());
    await settle(() =>
      storedValues().patchSnapshot({
        ...getDefaultGenerateSettings(ANIMA as never),
        animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER as never,
        animaInpaintAdapterWeight: 0.6,
        modelKey: ANIMA.key,
      })
    );

    expect(adapterSwitch()?.checked).toBe(true);
    expect(host!.textContent).not.toContain(offHelp);
    expect(sliderNamed(weightLabel).getAttribute('aria-disabled')).toBeNull();

    await userEvent.click(adapterSwitch()!.labels![0]!);
    await settle(flushGenerateDrafts);

    expect(storedValues().getSnapshot()).toMatchObject({
      animaInpaintAdapterEnabled: false,
      animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER,
      animaInpaintAdapterWeight: 0.6,
    });
    expect(adapterSwitch()?.checked).toBe(false);
    expect(host!.textContent).toContain(offHelp);
    expect(adapterSlot()?.getAttribute('data-component-value')).toBe(ANIMA_INPAINT_ADAPTER.key);
    expect(sliderNamed(weightLabel).getAttribute('aria-disabled')).toBe('true');

    await userEvent.click(adapterSwitch()!.labels![0]!);
    await settle(flushGenerateDrafts);

    expect(storedValues().getSnapshot()).toMatchObject({
      animaInpaintAdapterEnabled: true,
      animaInpaintAdapterModel: ANIMA_INPAINT_ADAPTER,
      animaInpaintAdapterWeight: 0.6,
    });
    expect(sliderNamed(weightLabel).getAttribute('aria-disabled')).toBeNull();
  });
});

describe('GenerateSettingsForm Z-Image Qwen3 encoder', () => {
  const Z_IMAGE = { ...MODEL, base: 'z-image', format: 'gguf_quantized', key: 'z-image', name: 'Z-Image' };
  const encoder = (variant: string) =>
    ({ base: 'any', key: `qwen3-${variant}`, name: `Qwen3 ${variant}`, type: 'qwen3_encoder', variant }) as const;
  const QWEN3_4B = encoder('qwen3_4b');
  const QWEN3_8B = encoder('qwen3_8b');
  const encoderSlot = () => host!.querySelector<HTMLElement>('[data-component-types="qwen3_encoder"]');
  const encoderError = () => encoderSlot()?.closest('[data-scope="field"]')?.querySelector('[role="alert"]');

  it('keeps a saved 8B encoder visible with its reason until a 4B one replaces it', async () => {
    stableGroups.models.catalog = [MODEL, Z_IMAGE, QWEN3_4B, QWEN3_8B] as GenerationModelCatalogItem[];
    await renderAdapter(buildAdapter());
    await settle(() =>
      storedValues().patchSnapshot({
        ...getDefaultGenerateSettings(Z_IMAGE as never),
        modelKey: Z_IMAGE.key,
        qwen3EncoderModel: QWEN3_8B,
      })
    );

    expect(encoderSlot()?.getAttribute('data-component-value')).toBe(QWEN3_8B.key);
    expect(encoderError()?.textContent).toBe(
      'The selected Qwen3 Encoder is not the 4B one Z-Image conditions on. Select a Qwen3 4B Encoder or clear it.'
    );
    expect(
      [...(encoderSlot()?.querySelectorAll('[data-component-pick]') ?? [])].map((pick) =>
        pick.getAttribute('data-component-pick')
      )
    ).toEqual([QWEN3_4B.key]);

    await settle(() => host!.querySelector<HTMLElement>(`[data-component-pick="${QWEN3_4B.key}"]`)?.click());
    await settle(flushGenerateDrafts);

    expect(storedValues().getSnapshot()).toMatchObject({ qwen3EncoderModel: { key: QWEN3_4B.key } });
    expect(encoderSlot()?.getAttribute('data-component-value')).toBe(QWEN3_4B.key);
    expect(encoderError()).toBeNull();
  });
});

describe('GenerateSettingsForm required components', () => {
  const krea2 = (format: string) =>
    ({ ...MODEL, base: 'krea-2', format, key: `krea-2-${format}`, name: 'Krea-2' }) as GenerationModelCatalogItem;
  const slotStates = () =>
    [
      ...host!.querySelectorAll<HTMLElement>('[data-component-types="vae"], [data-component-types="qwen3_vl_encoder"]'),
    ].map((slot) => ({
      error: slot.closest('[data-scope="field"]')?.querySelector('[role="alert"]')?.textContent ?? null,
      invalid: slot.getAttribute('aria-invalid') === 'true',
      placeholder: slot.getAttribute('data-component-placeholder'),
    }));
  const selectModel = (model: GenerationModelCatalogItem) =>
    settle(() => storedValues().patchSnapshot({ ...getDefaultGenerateSettings(model as never), modelKey: model.key }));

  it('asks for the components a checkpoint needs and defers to the model when it bundles them', async () => {
    const checkpoint = krea2('checkpoint');
    const diffusers = krea2('diffusers');
    stableGroups.models.catalog = [MODEL, checkpoint, diffusers];
    await renderAdapter(buildAdapter());

    await selectModel(checkpoint);
    expect(slotStates()).toEqual([
      {
        error: 'Generate needs a VAE for non-Diffusers Krea-2 models.',
        invalid: true,
        placeholder: 'widgets.generate.selectComponent',
      },
      {
        error: 'Generate needs a Qwen3-VL Encoder for non-Diffusers Krea-2 models.',
        invalid: true,
        placeholder: 'widgets.generate.selectComponent',
      },
    ]);

    await selectModel(diffusers);
    expect(slotStates()).toEqual([
      { error: null, invalid: false, placeholder: 'widgets.generate.modelDefault' },
      { error: null, invalid: false, placeholder: 'widgets.generate.modelDefault' },
    ]);
  });
});

describe('GenerateSettingsForm render isolation', () => {
  it('preserves a settled weight draft when removing another concept from an already open menu', async () => {
    const first: LoraModelConfig = { base: 'sdxl', key: 'ink', name: 'Ink', type: 'lora' };
    const second: LoraModelConfig = { base: 'sdxl', key: 'chalk', name: 'Chalk', type: 'lora' };
    await settle(() =>
      storedValues().patchSnapshot({
        loras: [first, second].map((model) => ({ isEnabled: true, model, weight: 0.75 })),
      })
    );
    const row = (name: string) => host!.querySelector<HTMLElement>(`[role="group"][aria-label="${name}"]`)!;

    // Capture the second row's menu before the first row's 250 ms draft has committed.
    await settle(() => {
      row('Ink')
        .querySelector('[role="slider"]')!
        .dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
      row('Chalk')
        .querySelector('[data-list-primary]')!
        .dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => (storedValues().getSnapshot().loras as GenerateLora[])[0]?.weight).toBe(0.8);
    const remove = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
      (item) => item.textContent === 'widgets.generate.conceptMenu.remove'
    );
    expect(remove).toBeDefined();
    await act(() => userEvent.click(remove!));

    expect(storedValues().getSnapshot().loras).toEqual([{ isEnabled: true, model: first, weight: 0.8 }]);
  });

  it('re-renders only the owning section while scrubbing, and nothing when the debounce persists it', async () => {
    const scrubber = stepsScrubber();
    const rect = scrubber.getBoundingClientRect();
    const startX = rect.left + rect.width * 0.3;

    await settle(() =>
      scrubber.dispatchEvent(
        new PointerEvent('pointerdown', { bubbles: true, button: 0, clientX: startX, pointerId: 1 })
      )
    );

    // Any pointerdown re-renders the prompt fields through their dnd context subscription; count the drag itself.
    commits.clear();

    for (let tick = 1; tick <= 10; tick += 1) {
      await settle(() =>
        window.dispatchEvent(
          new PointerEvent('pointermove', { bubbles: true, clientX: startX + tick * 4, pointerId: 1 })
        )
      );
    }

    await settle(() => window.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 })));

    const scrubbedSteps = stepsValue();

    expect(scrubbedSteps).toBeGreaterThan(30);
    // The debounce has not fired, so nothing is persisted yet.
    expect(storedValues().getSnapshot().steps).toBe(30);
    expect(committedSections()).toEqual(['render']);
    expect(commits.get('render')).toBeGreaterThanOrEqual(10);

    commits.clear();
    await settle(noop, 400);

    expect(storedValues().getSnapshot().steps).toBe(scrubbedSteps);
    expect(stepsValue()).toBe(scrubbedSteps);
    expect(committedSections()).toEqual([]);
  });

  it('shows stored values changed outside the form in the sections that read them', async () => {
    // A recall or another tab writes the store directly, bypassing the draft.
    await settle(() => storedValues().patchSnapshot({ steps: 12 }));

    expect(stepsValue()).toBe(12);
    expect(committedSections()).toEqual(['render']);
    for (const section of UNRELATED_SECTIONS) {
      expect(commits.has(section)).toBe(false);
    }
  });

  it('commits a pending edit to its own project when drafts are flushed ahead of a project change', async () => {
    await stepSteps();

    expect(patches).toEqual([]);

    // The Workbench's project commands flush drafts before they change the active project.
    await settle(flushGenerateDrafts);
    await renderAdapter(buildAdapter('project-2'));
    await settle(noop, 400);

    expect(patches).toEqual([{ patch: { steps: 31 }, projectId: 'project-1' }]);
    expect(storedValues('project-2').getSnapshot().steps).toBe(50);
    expect(stepsValue()).toBe(50);
  });

  it('discards a pending concept weight when another project finishes opening', async () => {
    const model: LoraModelConfig = { base: 'sdxl', key: 'ink-wash', name: 'Ink Wash', type: 'lora' };
    stableGroups.models.catalog = [MODEL, model];
    storedValues().patchSnapshot({ loras: [{ isEnabled: true, model, weight: 0.75 }] });
    storedValues('project-2').patchSnapshot({ loras: [{ isEnabled: true, model, weight: 1.25 }] });
    await renderAdapter(buildAdapter());
    const weight = () => host!.querySelector('[role="group"][aria-label="Ink Wash"] [role="slider"]');

    await settle(() => weight()?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })));
    expect(weight()?.getAttribute('aria-valuenow')).toBe('0.8');
    expect(patches).toEqual([]);

    // The open action already flushed before its server request; edits made during that request must not leak.
    await renderAdapter(buildAdapter('project-2'));
    await settle(noop, 400);

    expect(patches).toEqual([]);
    expect(storedValues('project-2').getSnapshot().loras).toEqual([{ isEnabled: true, model, weight: 1.25 }]);
    expect(weight()?.getAttribute('aria-valuenow')).toBe('1.25');
  });

  it('flushes a pending edit to its own project on unmount, not on ordinary re-renders', async () => {
    await stepSteps();
    // A re-render with a new adapter value but the same patch port, as a gallery or queue change causes.
    await renderAdapter(buildAdapter());
    await settle(() => storedValues().patchSnapshot({ positivePrompt: 'a harbour' }));

    expect(patches).toEqual([]);

    await act(() => root?.unmount());
    root = null;

    expect(patches).toEqual([{ patch: { steps: 31 }, projectId: 'project-1' }]);
  });

  it('applies a size reset at once, superseding a pending size edit that can then never land after it', async () => {
    await settle(() =>
      storedValues().patchSnapshot({
        aspectRatioId: '2:3',
        aspectRatioIsLocked: true,
        aspectRatioValue: 2 / 3,
        height: 1216,
        width: 832,
      })
    );
    await settle(() =>
      host!.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.resetSize"]')?.click()
    );
    const menu = document.querySelector<HTMLElement>('[role="menu"]');
    const highlighted = () => menu?.querySelector('[role="menuitemradio"][data-highlighted]')?.textContent ?? '';
    // Arrows down to Square at native size by its name, wherever the menu lists it.
    for (let step = 0; step < 3 && !highlighted().startsWith('widgets.generate.resetSizeSquareNative'); step += 1) {
      await settle(() => menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' })));
    }
    expect(highlighted()).toMatch(/^widgets\.generate\.resetSizeSquareNative/);

    // A width step still inside its debounce window, then the reset.
    await settle(() => {
      sliderNamed('widgets.generate.width').dispatchEvent(
        new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' })
      );
      menu?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
    });
    await settle(noop, 400);

    // SDXL is native at 1024.
    expect(patches).toEqual([
      { patch: { aspectRatioId: '1:1', aspectRatioValue: 1, height: 1024, width: 1024 }, projectId: 'project-1' },
    ]);
    expect(storedValues().getSnapshot()).toMatchObject({ aspectRatioIsLocked: true, height: 1024, width: 1024 });
  });

  it('batches fast edits to several fields into one patch on the project they were made in', async () => {
    await stepSteps();
    await stepSteps();
    await pressRight(sliderNamed('CFG'));

    expect(patches).toEqual([]);

    await settle(flushGenerateDrafts);
    await renderAdapter(buildAdapter('project-2'));
    await settle(noop, 400);

    expect(patches).toEqual([{ patch: { cfgScale: 7.5, steps: 32 }, projectId: 'project-1' }]);
    expect(storedValues('project-2').getSnapshot()).toMatchObject({ cfgScale: 7, steps: 50 });
  });

  it('keeps a pending field over an external update and applies the update to every other field', async () => {
    await stepSteps();
    await settle(() => storedValues().patchSnapshot({ cfgScale: 4, steps: 12 }));

    expect(stepsValue()).toBe(31);
    expect(sliderNamed('CFG').getAttribute('aria-valuenow')).toBe('4');

    await settle(noop, 400);

    expect(patches).toEqual([{ patch: { steps: 31 }, projectId: 'project-1' }]);
    expect(storedValues().getSnapshot()).toMatchObject({ cfgScale: 4, steps: 31 });
  });

  it('makes every pending draft readable from the store as soon as drafts are flushed', async () => {
    await showConcepts();
    await stepSteps();
    await pressRight(conceptRow('Ink')?.querySelector('[role="slider"]'));
    await pressRight(conceptRow('Chalk')?.querySelector('[role="slider"]'));

    expect(patches).toEqual([]);

    // Invoke and project export flush, then read the stored values synchronously.
    act(() => flushGenerateDrafts());

    expect(storedValues().getSnapshot()).toMatchObject({
      loras: [
        { model: ink, weight: 0.8 },
        { model: chalk, weight: 0.8 },
      ],
      steps: 31,
    });

    const flushed = patches.length;
    await settle(noop, 400);

    expect(patches).toHaveLength(flushed);
  });

  it('lands two concept weight drafts made together without either overwriting the other', async () => {
    await showConcepts();
    await pressRight(conceptRow('Ink')?.querySelector('[role="slider"]'));
    await pressRight(conceptRow('Chalk')?.querySelector('[role="slider"]'));
    await pressRight(conceptRow('Chalk')?.querySelector('[role="slider"]'));

    await expect
      .poll(() => storedValues().getSnapshot().loras)
      .toEqual([
        { isEnabled: true, model: ink, weight: 0.8 },
        { isEnabled: true, model: chalk, weight: 0.85 },
      ]);
    expect(conceptRow('Ink')?.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.8');
    expect(conceptRow('Chalk')?.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.85');
  });

  it('removes a concept whose weight draft is still pending without restoring it', async () => {
    await showConcepts();
    await pressRight(conceptRow('Ink')?.querySelector('[role="slider"]'));

    await settle(() =>
      conceptRow('Ink')
        ?.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.removeConceptNamed"]')
        ?.click()
    );
    await settle(noop, 400);

    expect(conceptRow('Ink')).toBeNull();
    expect(storedValues().getSnapshot().loras).toEqual([{ isEnabled: true, model: chalk, weight: 0.75 }]);
  });

  // Documents the account boundary: the unmounting form settles its edit through the account that made it.
  it('settles a pending edit through its own account, never the next one', async () => {
    await stepSteps();
    projectValues.set('next-account-project', createExternalStoreCore<Record<string, unknown>>(INITIAL_VALUES));
    const nextAccountPatches: Array<Record<string, unknown>> = [];
    const nextAccountAdapter = {
      ...buildAdapter('next-account-project'),
      project: {
        activeProjectId: 'next-account-project',
        invocationSourceId: 'generate',
        showPromptSyntaxHighlighting: false,
      },
      settings: { patchGenerateSettings: (patch: Record<string, unknown>) => nextAccountPatches.push(patch) },
    } as unknown as GenerationUiAdapter;

    accountLifecycle.invalidate();
    await renderAdapter(nextAccountAdapter, 'account-2');
    // Capabilities are account-owned, so the next account's form waits for them again.
    await settle(ensureArchitectureCapabilitiesLoaded, 400);

    // The App disposes the first account's store, so this patch never persists past the switch.
    expect(patches).toEqual([{ patch: { steps: 31 }, projectId: 'project-1' }]);
    expect(nextAccountPatches).toEqual([]);
    expect(storedValues('next-account-project').getSnapshot().steps).toBe(30);
    expect(stepsValue()).toBe(30);
  });

  it('switches models from the draft, carrying a pending edit in one commit', async () => {
    stableGroups.models.catalog = [MODEL, OTHER_SDXL];
    await renderAdapter(buildAdapter());
    await stepSteps();

    await settle(() => host!.querySelector<HTMLElement>('[data-model-pick="sdxl-2"]')?.click());

    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(patches).toHaveLength(1);
    expect(patches[0]).toMatchObject({ patch: { modelKey: 'sdxl-2', steps: 31 }, projectId: 'project-1' });

    await settle(noop, 400);

    expect(patches).toHaveLength(1);
    expect(storedValues().getSnapshot()).toMatchObject({ modelKey: 'sdxl-2', steps: 31 });
    expect(stepsValue()).toBe(31);
  });

  it('switches at once when nothing is lost, with an Undo that restores the earlier model and size', async () => {
    const undoable = vi.fn();
    stableGroups.notifications.undoable = undoable;
    stableGroups.models.catalog = [MODEL, FLUX, ink];
    storedValues().patchSnapshot({ loras: [{ isEnabled: true, model: ink, weight: 0.75 }] });
    await renderAdapter(buildAdapter());
    // 1032 sits on SDXL's 8 px grid but not FLUX's 16 px one.
    await pressRight(sliderNamed('widgets.generate.width'));

    await settle(() => host!.querySelector<HTMLElement>('[data-model-pick="flux"]')?.click());

    // The SDXL concept is set aside, not lost, so the switch needs no confirmation.
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
    expect(patches.at(-1)?.patch).toMatchObject({ modelKey: 'flux' });
    expect((patches.at(-1)?.patch.width as number) % 16).toBe(0);
    expect(storedValues().getSnapshot().loras).toEqual([{ isEnabled: true, model: ink, weight: 0.75 }]);
    expect(undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.switchedModel', {
      message:
        'widgets.generate.switchChanges.turnedOff: widgets.generate.switchSubjects.concepts widgets.generate.switchChanges.size',
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-settings:project-1',
    });

    await settle(() => undoable.mock.calls[0]![1].onUndo());

    expect(storedValues().getSnapshot()).toMatchObject({ modelKey: 'sdxl', width: 1032 });
  });

  it('confirms a switch that removes something, naming what goes and what is only turned off', async () => {
    stableGroups.models.catalog = [MODEL, FLUX, ink, SDXL_VAE];
    storedValues().patchSnapshot({ loras: [{ isEnabled: true, model: ink, weight: 0.75 }], vae: SDXL_VAE });
    await renderAdapter(buildAdapter());
    await stepSteps();
    // 1032 sits on SDXL's 8 px grid but not FLUX's 16 px one; only the draft holds it.
    await pressRight(sliderNamed('widgets.generate.width'));

    await settle(() => host!.querySelector<HTMLElement>('[data-model-pick="flux"]')?.click());

    const dialog = document.querySelector<HTMLElement>('[role="alertdialog"]');
    const removed = 'widgets.generate.switchModelLines.removed: widgets.generate.switchSubjects.vae';
    expect(dialog?.textContent).toContain(removed);
    expect(dialog?.textContent).toContain(
      'widgets.generate.switchModelLines.turnedOff: widgets.generate.switchSubjects.concepts'
    );
    // The size moves to the FLUX grid, which the dialog reports beside the loss rather than as one.
    expect(dialog?.textContent).toContain('widgets.generate.switchChanges.size');
    expect(patches).toEqual([]);

    const confirm = [...dialog!.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'widgets.generate.switchModelConfirm'
    );
    const frames = closingFrames(await recordDialogExit(dialog!, () => settle(() => confirm?.click())));

    // The switch lands as the dialog closes, and the dialog animates out still listing what was confirmed.
    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain(removed);
    }

    await settle(noop, 400);

    expect(patches).toHaveLength(1);
    expect(patches[0]?.patch).toMatchObject({ modelKey: 'flux', steps: 31, vae: null });
    expect((patches[0]?.patch.width as number) % 16).toBe(0);
    expect(storedValues().getSnapshot()).toMatchObject({ modelKey: 'flux', steps: 31, vae: null });
  });

  it('confirms a switch another surface asked for, the way its own picker does', async () => {
    stableGroups.models.catalog = [MODEL, FLUX, SDXL_VAE];
    storedValues().patchSnapshot({ vae: SDXL_VAE });
    await renderAdapter(buildAdapter());

    await settle(() =>
      requestGenerateModelSwitchConfirmation({ model: FLUX as GenerateModelConfig, projectId: 'project-2' })
    );
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();

    await settle(() =>
      requestGenerateModelSwitchConfirmation({ model: FLUX as GenerateModelConfig, projectId: 'project-1' })
    );
    const dialog = document.querySelector<HTMLElement>('[role="alertdialog"]');
    expect(dialog?.textContent).toContain('widgets.generate.switchSubjects.vae');

    await settle(() =>
      [...dialog!.querySelectorAll<HTMLButtonElement>('button')]
        .find((button) => button.textContent === 'widgets.generate.switchModelConfirm')
        ?.click()
    );
    await settle(noop, 400);

    expect(storedValues().getSnapshot()).toMatchObject({ modelKey: 'flux', vae: null });
  });

  it('keeps a model that is no longer installed, names it, and offers another pick or the Model Manager', async () => {
    const gone = { ...MODEL, key: 'gone', name: 'Gone XL' };
    const openManager = vi.fn();
    stableGroups.models.openManager = openManager;
    mainPickerOpened.mockClear();
    projectValues.set(
      'project-1',
      createExternalStoreCore<Record<string, unknown>>({ ...INITIAL_VALUES, model: gone, modelKey: gone.key })
    );
    await renderAdapter(buildAdapter());

    expect(storedValues().getSnapshot().modelKey).toBe('gone');
    expect(host?.textContent).toContain('widgets.generate.issues.modelNotInstalled');
    expect(patches).toEqual([]);

    const button = (name: string) =>
      [...host!.querySelectorAll<HTMLButtonElement>('button')].find((candidate) => candidate.textContent === name);

    await settle(() => button('widgets.generate.chooseAnotherModel')?.click());
    expect(mainPickerOpened).toHaveBeenCalledOnce();

    await settle(() => button('widgets.generate.openModelManager')?.click());
    expect(openManager).toHaveBeenCalledExactlyOnceWith({ modelType: 'main' });
  });

  it('badges a section holding an issue, and its one-step fix patches the setting and moves focus on', async () => {
    projectValues.set(
      'project-1',
      createExternalStoreCore<Record<string, unknown>>({ ...INITIAL_VALUES, aspectRatioId: 'Free', width: 1001 })
    );
    await renderAdapter(buildAdapter());
    const sizeSection = () =>
      [...host!.querySelectorAll<HTMLElement>('.generate-section')].find((section) =>
        section.textContent?.includes('widgets.generate.size')
      )!;

    expect(sizeSection().querySelector('[data-part="trigger"]')?.textContent).toContain('widgets.generate.issueCount');

    const fix = sizeSection().querySelector<HTMLButtonElement>('[data-issue-fix]')!;
    expect(fix.textContent).toBe('widgets.generate.issueFixes.setSize');
    await settle(() => fix.click(), 50);

    expect(patches.at(-1)).toMatchObject({
      patch: { aspectRatioValue: 1000 / 1024, width: 1000 },
      projectId: 'project-1',
    });
    expect(sizeSection().querySelector('[data-part="trigger"]')?.textContent).not.toContain(
      'widgets.generate.issueCount'
    );
    expect(document.activeElement).toBe(sizeSection().querySelector('[data-part="trigger"]'));
  });
});

describe('GenerateSettingsForm model loading', () => {
  it('reads the models again from Retry when they could not be loaded, and shows it working', async () => {
    let finish!: () => void;
    const refresh = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        })
    );
    stableGroups.models = { ...stableGroups.models, catalog: [], error: 'Server said no', refresh, status: 'loaded' };
    await renderAdapter(buildAdapter());

    expect(host!.textContent).toContain('widgets.generate.modelsCouldNotLoad');
    expect(host!.textContent).toContain('Server said no');
    const retry = [...host!.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'common.retry'
    )!;

    await settle(() => retry.click());
    expect(refresh).toHaveBeenCalledOnce();
    expect(retry.getAttribute('aria-busy')).toBe('true');

    await settle(() => finish());
    expect(retry.hasAttribute('aria-busy')).toBe(false);
  });
});

const CINEMATIC = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, 35mm' };

describe('GenerateSettingsForm prompt changes past the prompt text', () => {
  const buttonNamed = (name: string) =>
    [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.getAttribute('aria-label') === name
    )!;

  it('uses a history entry that replaces the negative prompt, with an Undo that keeps typing since', async () => {
    const undoable = vi.fn();
    stableGroups.notifications.undoable = undoable;
    stableGroups.promptHistory.items = [{ negativePrompt: 'grainy', positivePrompt: 'a fox in snow' }];
    storedValues().patchSnapshot({ negativePrompt: 'blurry' });
    await renderAdapter(buildAdapter());

    await settle(() => buttonNamed('widgets.generate.promptHistory').click());
    await settle(() => buttonNamed('widgets.generate.usePrompt').click());

    expect(storedValues().getSnapshot()).toMatchObject({ negativePrompt: 'grainy', positivePrompt: 'a fox in snow' });
    expect(undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.promptHistoryUsed', {
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-prompt:project-1',
    });

    // The positive prompt was edited since; the Undo leaves that edit alone.
    await settle(() => storedValues().patchSnapshot({ positivePrompt: 'a fox in deep snow' }));
    await settle(() => undoable.mock.calls[0]![1].onUndo());

    expect(storedValues().getSnapshot()).toMatchObject({
      negativePrompt: 'blurry',
      positivePrompt: 'a fox in deep snow',
    });
  });

  it('applies a template chosen in the panel, with an Undo that puts back the one before', async () => {
    const undoable = vi.fn();
    const previous = { id: 'noir', name: 'Noir', negativePrompt: '', positivePrompt: '{prompt}, noir' };
    stableGroups.notifications.undoable = undoable;
    storedValues().patchSnapshot({ promptTemplate: previous });
    queryClient.setQueryData(promptTemplatesQueryOptions().queryKey, [
      { ...CINEMATIC, hasImage: false, isDefault: true, isPublic: true, userId: 'system' },
    ]);
    await renderAdapter(buildAdapter());

    await settle(() => buttonNamed('widgets.generate.promptTemplates.title').click(), 50);
    await settle(
      () =>
        [...document.querySelectorAll<HTMLButtonElement>('[data-scope="popover"] button')]
          .find((button) => button.textContent?.includes('Cinematic'))!
          .click(),
      50
    );

    expect(storedValues().getSnapshot().promptTemplate).toEqual(CINEMATIC);
    expect(undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.promptTemplateApplied', {
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-prompt:project-1',
    });

    await settle(() => undoable.mock.calls[0]![1].onUndo());

    expect(storedValues().getSnapshot().promptTemplate).toEqual(previous);
  });

  it('merges a template into the prompt with an Undo that is all or nothing', async () => {
    const undoable = vi.fn();
    const info = vi.fn();
    const template = { ...CINEMATIC, negativePrompt: '{prompt}, blurry' };
    stableGroups.notifications = { ...stableGroups.notifications, info, undoable };
    storedValues().patchSnapshot({
      negativePrompt: 'grainy',
      positivePrompt: 'a {red|blue} cat',
      promptTemplate: template,
    });
    queryClient.setQueryData(
      dynamicPromptsKeys.expansion({
        combinatorial: true,
        max_prompts: 100,
        prompt: 'a {red|blue} cat, 35mm',
        seed: null,
      }),
      { error: null, prompts: ['a red cat, 35mm', 'a blue cat, 35mm'] }
    );
    await renderAdapter(buildAdapter());

    await settle(() => buttonNamed('widgets.generate.dynamicPrompts.showPrompts').click(), 700);
    await settle(
      () =>
        [...document.querySelectorAll<HTMLButtonElement>('[data-scope="popover"] button')]
          .find((button) => button.textContent?.includes('a red cat, 35mm'))!
          .click(),
      50
    );

    expect(storedValues().getSnapshot()).toMatchObject({
      negativePrompt: 'grainy, blurry',
      positivePrompt: 'a red cat, 35mm',
      promptTemplate: null,
    });
    expect(undoable).toHaveBeenCalledWith('widgets.generate.promptTemplateMerged', expect.anything());
    const undo = undoable.mock.calls.at(-1)![1].onUndo as () => void;

    // Half a merge would be a prompt nobody wrote: with the positive prompt edited since, nothing is undone.
    await settle(() => storedValues().patchSnapshot({ positivePrompt: 'a red cat, 50mm' }));
    await settle(undo);
    expect(storedValues().getSnapshot()).toMatchObject({ negativePrompt: 'grainy, blurry', promptTemplate: null });
    expect(info).toHaveBeenCalledWith('widgets.generate.undoStale');

    await settle(() => storedValues().patchSnapshot({ positivePrompt: 'a red cat, 35mm' }));
    await settle(undo);
    expect(storedValues().getSnapshot()).toMatchObject({
      negativePrompt: 'grainy',
      positivePrompt: 'a {red|blue} cat',
      promptTemplate: template,
    });
  });

  it('stops using a template from the Templates panel, with an Undo that applies it again', async () => {
    const undoable = vi.fn();
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, 35mm' };
    stableGroups.notifications.undoable = undoable;
    storedValues().patchSnapshot({ promptTemplate: template });
    await renderAdapter(buildAdapter());

    await settle(() => buttonNamed('widgets.generate.promptTemplates.title').click(), 50);
    await settle(
      () =>
        [...document.querySelectorAll<HTMLButtonElement>('[data-scope="popover"] button')]
          .find((button) => button.textContent === 'widgets.generate.promptTemplates.clearApplied')!
          .click(),
      50
    );

    expect(storedValues().getSnapshot().promptTemplate).toBeNull();
    expect(undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.promptTemplateCleared', {
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-prompt:project-1',
    });

    await settle(() => undoable.mock.calls[0]![1].onUndo());

    expect(storedValues().getSnapshot().promptTemplate).toEqual(template);
  });
});
