/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GenerationModelCatalogItem } from '@features/generation/contracts';
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
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GenerateWidgetView } from './GenerateWidgetView';
import { GenerationUiProvider, type GenerationUiAdapter } from './GenerationUiContext';

vi.mock('react-i18next', () => {
  const t = (key: string) => key;
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

/** Groups shared by every adapter a test renders, as the app keeps them stable across renders. */
const createStableGroups = () => ({
  CanvasGenerationSections: () => null,
  account: { currentUserId: null, multiuserEnabled: false },
  capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
  gallery: { findImage: noop, selectedImage: null, touchImages: noop },
  models: {
    ModelSelect: () => null,
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
    queueInsights: { secondsPerRun: null, seedHistory: [] },
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

const committedSections = () => [...commits.keys()].sort();

const renderAdapter = (adapter: GenerationUiAdapter) =>
  settle(() =>
    root?.render(
      <QueryClientProvider client={queryClient}>
        <ChakraProvider value={system}>
          <DndContext>
            <GenerationUiProvider adapter={adapter}>
              <GenerateWidgetView />
            </GenerationUiProvider>
          </DndContext>
        </ChakraProvider>
      </QueryClientProvider>
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

  it('flushes a pending edit to its own project when the app switches projects before the debounce', async () => {
    await stepSteps();

    expect(patches).toEqual([]);

    // Project switches flush drafts first, then activate the other project.
    await settle(flushGenerateDrafts);
    await renderAdapter(buildAdapter('project-2'));
    await settle(noop, 400);

    expect(patches).toEqual([{ patch: { steps: 31 }, projectId: 'project-1' }]);
    expect(storedValues('project-2').getSnapshot().steps).toBe(50);
    expect(stepsValue()).toBe(50);
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
});
