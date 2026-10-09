/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
import type { GenerateLora, LoraModelConfig } from '@features/generation/core/types';
import type { ComponentType } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { ensureArchitectureCapabilitiesLoaded } from '@features/generation/data/architectureCapabilitiesStore';
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
const CATALOG: readonly GenerationModelCatalogItem[] = [MODEL];
const OTHER_SDXL = { ...MODEL, key: 'sdxl-2', name: 'SDXL 2' } as GenerationModelCatalogItem;
// A coarser size grid than SDXL's, and no SDXL LoRAs.
const FLUX = { ...MODEL, base: 'flux', key: 'flux', name: 'FLUX' } as GenerationModelCatalogItem;

/** The model card's picker, reduced to buttons that hand a model to its real selection handler. */
const MainModelPicker = ({ modelTypes, onChange }: GenerationModelSelectProps) =>
  modelTypes.includes('external_image_generator')
    ? [OTHER_SDXL, FLUX].map((model) => (
        <button key={model.key} data-model-pick={model.key} type="button" onClick={() => onChange(model)}>
          {model.name}
        </button>
      ))
    : null;
const INITIAL_VALUES: Record<string, unknown> = {
  ...getDefaultGenerateSettings(MODEL as never),
  modelKey: MODEL.key,
  positivePrompt: 'a lighthouse',
  steps: 30,
};
const UNRELATED_SECTIONS = ['model', 'prompts', 'dimensions', 'guidance', 'components', 'advanced'];

const noop = () => undefined;
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
  capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
  gallery: { findImage: noop, selectedImage: null, touchImages: noop },
  models: {
    ModelSelect: MainModelPicker,
    catalog: CATALOG,
    ensureLoaded: noop,
    error: null,
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
    openManager: noop,
    status: 'loaded',
  },
  notifications: { error: noop, info: noop, reportError: noop },
  presets: { presets: [], remove: noop, rename: noop, save: noop },
  promptHistory: { clear: noop, items: [], remove: noop },
  queueInsights: { getSnapshot: () => NO_QUEUE_INSIGHTS, subscribe: () => noop },
  rebalancePresets: { presets: [], remove: noop, rename: noop, save: noop },
  sectionPreferences: { sectionsOpen: ALL_SECTIONS_OPEN, setSectionOpen: noop },
  settings: {
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

const ink: LoraModelConfig = { base: 'sdxl', key: 'ink', name: 'Ink', type: 'lora' };
const chalk: LoraModelConfig = { base: 'sdxl', key: 'chalk', name: 'Chalk', type: 'lora' };

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

  it('confirms a lossy switch against the draft, pending edits included', async () => {
    stableGroups.models.catalog = [MODEL, FLUX, ink];
    storedValues().patchSnapshot({ loras: [{ isEnabled: true, model: ink, weight: 0.75 }] });
    await renderAdapter(buildAdapter());
    await stepSteps();
    // 1032 sits on SDXL's 8 px grid but not FLUX's 16 px one; only the draft holds it.
    await pressRight(sliderNamed('widgets.generate.width'));

    await settle(() => host!.querySelector<HTMLElement>('[data-model-pick="flux"]')?.click());

    const dialog = document.querySelector<HTMLElement>('[role="alertdialog"]');
    expect(dialog?.textContent).toContain('widgets.generate.switchModelBody: Dimensions and LoRAs');
    expect(patches).toEqual([]);

    const confirm = [...dialog!.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'widgets.generate.switchModelConfirm'
    );
    const frames = closingFrames(await recordDialogExit(dialog!, () => settle(() => confirm?.click())));

    // The switch lands as the dialog closes, and the dialog animates out still listing what was confirmed.
    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain('widgets.generate.switchModelBody: Dimensions and LoRAs');
    }

    await settle(noop, 400);

    expect(patches).toHaveLength(1);
    expect(patches[0]?.patch).toMatchObject({ loras: [], modelKey: 'flux', steps: 31 });
    expect((patches[0]?.patch.width as number) % 16).toBe(0);
    expect(storedValues().getSnapshot()).toMatchObject({ loras: [], modelKey: 'flux', steps: 31 });
  });
});
