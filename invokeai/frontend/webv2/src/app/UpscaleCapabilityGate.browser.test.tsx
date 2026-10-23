import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type { ModelConfig } from '@features/models';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import { createDefaultUpscaleWidgetValues } from '@features/upscale';
import { UpscaleWidgetView } from '@features/upscale/ui/UpscaleWidgetView';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { UpscaleUiAdapterProvider } from './UpscaleUiAdapter';

const modelSnapshot = {
  models: [
    { base: 'z-image', format: 'diffusers', hash: 'z', key: 'z-turbo', name: 'Z-Image Turbo', type: 'main' },
    {
      base: 'any',
      format: 'checkpoint',
      hash: 's',
      key: 'spandrel',
      name: 'Spandrel',
      scale: 4,
      type: 'spandrel_image_to_image',
    },
    // Installed before Invoke recorded native scales, and not re-identified since.
    { base: 'any', format: 'checkpoint', hash: 'l', key: 'legacy', name: 'Legacy', type: 'spandrel_image_to_image' },
  ] as ModelConfig[],
  status: 'loaded',
};
const getArchitectureCapabilities = vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>();
// Null for a session that may not manage models, as the hook answers.
const modelManager = vi.hoisted(() => ({ open: null as ((modelKey: string) => void) | null }));
const noop = () => undefined;
const generationAdapter = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: noop },
} as unknown as GenerationUiAdapter;

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;

vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => getArchitectureCapabilities(),
}));
vi.mock('react-i18next', () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
  getModelImageUrl: () => '',
  useModelsSelector: (selector: (snapshot: typeof modelSnapshot) => unknown) => selector(modelSnapshot),
  useOpenModelInManager: () => modelManager.open,
}));
// The pickers' transport is not under test; the form around them is. The stand-in takes its description from the
// enclosing field, as ModelSelect's own tests show the picker does.
vi.mock('@features/models/react', async (importOriginal) => {
  const { useFieldContext } = await import('@chakra-ui/react');

  return {
    ...(await importOriginal<Record<string, unknown>>()),
    ModelSelect: () => (
      <button
        aria-describedby={useFieldContext()?.getInputProps()['aria-describedby']}
        aria-haspopup="listbox"
        aria-label="Model picker"
        type="button"
      />
    ),
  };
});
vi.mock('@features/gallery/mediaSlot', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryMediaSlot: () => null,
}));
vi.mock('@features/generation/components', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  NegativePromptField: () => null,
  PositivePromptField: () => null,
}));
vi.mock('@platform/ui/toaster', () => ({ toaster: { create: () => undefined } }));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkbenchPreferenceSelector: () => false,
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => noop }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => noop }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWorkbenchCommands: () => store.commands,
  useWorkbenchQueries: () => store.queries,
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const settle = async (run: () => void = noop) => {
  await act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 0);
    });
  });
};

const storedValues = () => getProjectWidgetValues(store.getSnapshot().activeProject, 'upscale');

beforeEach(() => {
  store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  getArchitectureCapabilities.mockReset();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  // Returns the capability store and the core registry to their unloaded state.
  accountLifecycle.invalidate();
});

const renderUpscale = () =>
  settle(() =>
    root.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={generationAdapter}>
            <UpscaleUiAdapterProvider>
              <UpscaleWidgetView />
            </UpscaleUiAdapterProvider>
          </GenerationUiProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );

describe('Upscale Refine and Model only modes', () => {
  const sectionLabels = () =>
    ['widgets.upscale.detailGuidance', 'widgets.upscale.generation', 'widgets.upscale.advanced'].filter((label) =>
      [...host.querySelectorAll('button')].some((button) => button.textContent?.startsWith(label))
    );
  const pickMode = (label: string) =>
    settle(() =>
      [...host.querySelectorAll<HTMLElement>('[data-part="item"]')].find((item) => item.textContent === label)?.click()
    );

  it('hides what only the detail pass reads in Model only, explains the native scale, and restores Refine as left', async () => {
    getArchitectureCapabilities.mockResolvedValue(architectureCapabilitiesFixture);
    store.commands.widgets.patchValues('upscale', {
      ...createDefaultUpscaleWidgetValues(),
      creativity: 5,
      inputImage: { height: 800, image_name: 'input.png', width: 1000 },
      scale: 2,
    });

    await renderUpscale();

    const modes = host.querySelector('[aria-label="widgets.upscale.mode"]');
    expect(modes?.querySelector('[data-state="checked"]')?.textContent).toBe('widgets.upscale.modeRefine');
    expect(sectionLabels()).toHaveLength(3);
    expect(host.querySelector('[aria-label="widgets.upscale.presetsLabel"]')).not.toBeNull();
    expect(host.textContent).not.toContain('widgets.upscale.nativeScale');

    await pickMode('widgets.upscale.modeModelOnly');

    expect(storedValues()).toMatchObject({ creativity: 5, mode: 'model-only' });
    expect(sectionLabels()).toEqual([]);
    expect(host.querySelector('[aria-label="widgets.upscale.presetsLabel"]')).toBeNull();
    expect(host.textContent).not.toContain('widgets.upscale.creativity');
    // The 4x upscaler runs once and resizes down to Scale 2.
    expect(host.textContent).toContain('widgets.upscale.nativeScale');
    expect(host.textContent).toContain('widgets.upscale.nativeScalePassesResize');
    expect(host.textContent).toContain('2,000 × 1,600');

    await pickMode('widgets.upscale.modeRefine');

    expect(storedValues()).toMatchObject({ creativity: 5, mode: 'refine' });
    expect(sectionLabels()).toHaveLength(3);
    expect(host.textContent).not.toContain('widgets.upscale.nativeScale');
  });
});

describe('Upscale without the served architecture table', () => {
  it('keeps Model only usable, since it reads no main-model capabilities, and holds back only Refine', async () => {
    store.commands.widgets.patchValues('upscale', {
      ...createDefaultUpscaleWidgetValues(),
      inputImage: { height: 800, image_name: 'input.png', width: 1000 },
      mode: 'model-only',
      upscaleModel: modelSnapshot.models[1] as never,
    });
    getArchitectureCapabilities.mockRejectedValue(new Error('Fixture capability outage.'));

    await renderUpscale();

    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(host.textContent).toContain('widgets.upscale.nativeScale');
    expect(host.textContent).toContain('4,000 × 3,200');

    await settle(() =>
      [...host.querySelectorAll<HTMLElement>('[data-part="item"]')]
        .find((item) => item.textContent === 'widgets.upscale.modeRefine')
        ?.click()
    );

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('widgets.upscale.capabilitiesLoadFailed');
    // The mode stays reachable above the gate, so the user can go back to Model only.
    expect(host.querySelector('[aria-label="widgets.upscale.mode"]')).not.toBeNull();
    expect(storedValues()).toMatchObject({ mode: 'refine', model: null });
  });

  it("picks no model until the table loads, then gives the one it picks that model's CFG", async () => {
    // A fresh widget: no model yet, the widget's CFG 2.
    store.commands.widgets.patchValues('upscale', { ...createDefaultUpscaleWidgetValues() });
    getArchitectureCapabilities.mockRejectedValueOnce(new Error('Fixture capability outage.'));

    await settle(() =>
      root.render(
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <GenerationUiProvider adapter={generationAdapter}>
              <UpscaleUiAdapterProvider>
                <UpscaleWidgetView />
              </UpscaleUiAdapterProvider>
            </GenerationUiProvider>
          </ChakraProvider>
        </QueryClientProvider>
      )
    );

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('widgets.upscale.capabilitiesLoadFailed');
    // Turbo picked now would keep CFG 2 for good: nothing re-applies a default to a selected model.
    expect(storedValues()).toMatchObject({ cfgScale: 2, model: null });

    getArchitectureCapabilities.mockResolvedValueOnce(architectureCapabilitiesFixture);
    await settle(() =>
      [...host.querySelectorAll('button')].find((button) => button.textContent === 'widgets.upscale.retry')?.click()
    );

    expect(host.querySelector('[role="alert"]')).toBeNull();
    // Turbo is distilled for CFG 1, the default the backend serves for it.
    expect(storedValues()).toMatchObject({ cfgScale: 1, model: { key: 'z-turbo' } });

    await settle(() =>
      [...host.querySelectorAll('button')]
        .find((button) => button.textContent === 'widgets.upscale.generation')
        ?.click()
    );

    // Euler and Heun both start where the blend left the latents on Z-Image; LCM does not.
    expect(host.textContent).toContain('widgets.upscale.cfgScale');
    const scheduler = host.querySelector<HTMLInputElement>('input[aria-label="widgets.upscale.scheduler"]');
    expect(scheduler?.value).toBe('Euler');

    await settle(() =>
      scheduler
        ?.closest('[data-scope="combobox"][data-part="root"]')
        ?.querySelector<HTMLButtonElement>('button[aria-label="common.openSelector"]')
        ?.click()
    );

    const offered = [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent);
    expect(offered).toEqual(['Euler', 'Heun (2nd order)']);
  });
});

describe('Upscale Model only with an upscaler of unknown native scale', () => {
  const NATIVE_SCALE_UNKNOWN = 'widgets.upscale.nativeScaleUnknown';
  const OPEN_IN_MODEL_MANAGER = 'widgets.upscale.openInModelManager';
  const renderWithLegacyUpscaler = (mode: 'model-only' | 'refine') => {
    getArchitectureCapabilities.mockResolvedValue(architectureCapabilitiesFixture);
    store.commands.widgets.patchValues('upscale', {
      ...createDefaultUpscaleWidgetValues(),
      inputImage: { height: 800, image_name: 'input.png', width: 1000 },
      mode,
      upscaleModel: modelSnapshot.models[2] as never,
    });

    return renderUpscale();
  };
  /** The Spandrel model picker's accessible description, read through its `aria-describedby`. */
  const upscalerDescription = () => {
    const pickers = [...host.querySelectorAll<HTMLElement>('[aria-haspopup="listbox"]')];
    const spandrel = pickers.find((picker) =>
      picker.closest('[data-part="root"]')?.textContent?.includes('widgets.upscale.spandrelModel')
    );

    return (spandrel?.getAttribute('aria-describedby') ?? '')
      .split(' ')
      .map((id) => document.getElementById(id)?.textContent ?? '')
      .join(' ');
  };
  const openButton = () =>
    [...host.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === OPEN_IN_MODEL_MANAGER
    );

  afterEach(() => {
    modelManager.open = null;
  });

  it('says why no plan is shown and opens that upscaler in the Model Manager to re-identify it', async () => {
    const open = vi.fn();
    modelManager.open = open;

    await renderWithLegacyUpscaler('model-only');

    expect(upscalerDescription()).toBe(NATIVE_SCALE_UNKNOWN);
    expect(host.textContent).not.toContain('widgets.upscale.nativeScaleValue');

    await settle(() => openButton()?.click());

    expect(open).toHaveBeenCalledWith('legacy');
  });

  it('keeps the hint without the link for a session that cannot manage models', async () => {
    await renderWithLegacyUpscaler('model-only');

    expect(upscalerDescription()).toBe(NATIVE_SCALE_UNKNOWN);
    expect(openButton()).toBeUndefined();
  });

  it('says nothing in Refine, whose detail pass resizes to Scale whatever the upscaler does', async () => {
    modelManager.open = vi.fn();

    await renderWithLegacyUpscaler('refine');

    expect(host.textContent).not.toContain(NATIVE_SCALE_UNKNOWN);
    expect(upscalerDescription()).toBe('widgets.upscale.spandrelModelHelp');
  });
});
