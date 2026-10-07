import type { ModelConfig } from '@features/models';
import type { VideoWidgetValues } from '@features/video';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { GenerationUiProvider, type GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import { getDefaultVideoSettings } from '@features/video';
import { VideoWidgetView } from '@features/video/ui/VideoWidgetView';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { VideoUiAdapterProvider } from './VideoUiAdapter';

/** Switching to a model that cannot keep the panel's setup asks first; the real store holds what each step wrote. */

const model = (key: string, base: string, type: string, name: string, extra: Partial<ModelConfig> = {}) =>
  ({
    base,
    file_size: 1,
    format: 'diffusers',
    hash: key,
    key,
    name,
    path: key,
    source: key,
    source_type: 'path',
    type,
    ...extra,
  }) as ModelConfig;

const ltxDev = model('ltx-dev', 'ltx-2', 'main', 'LTX-2 Dev', { format: 'checkpoint', variant: 'ltx2_dev' });
const ltxDistilled = model('ltx-distilled', 'ltx-2', 'main', 'LTX-2 Distilled', {
  format: 'checkpoint',
  variant: 'ltx2_distilled',
});
const wanT2v = model('wan-t2v', 'wan', 'main', 'Wan 2.2 T2V', { format: 'gguf_quantized', variant: 't2v_a14b' });
const inkWash = model('ink-wash', 'ltx-2', 'lora', 'Ink Wash');
const filmGrain = model('film-grain', 'ltx-2', 'lora', 'Film Grain');
const catalog = [
  ltxDev,
  ltxDistilled,
  wanT2v,
  model('ltx-components', 'ltx-2', 'main', 'LTX-2 Components', { components_only: true, variant: 'ltx2_dev' }),
  model('gemma', 'ltx-2', 'gemma4_encoder', 'Gemma-4'),
  inkWash,
  filmGrain,
];

const { modelStore, notify, pickerIds } = vi.hoisted(() => {
  let snapshot: { models: readonly unknown[]; status: string } = { models: [], status: 'loaded' };
  const listeners = new Set<() => void>();

  return {
    modelStore: {
      get: () => snapshot,
      set: (models: readonly unknown[]) => {
        snapshot = { models, status: 'loaded' };
        listeners.forEach((listener) => listener());
      },
      subscribe: (listener: () => void) => {
        listeners.add(listener);
        return () => void listeners.delete(listener);
      },
    },
    notify: vi.fn(),
    /** The id each main-model picker rendered with; it keys the picker's remembered view and filter. */
    pickerIds: [] as (string | undefined)[],
  };
});
const useModelSnapshot = () => useSyncExternalStore(modelStore.subscribe, modelStore.get);

vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
  getModelImageUrl: () => '',
  useModelsSelector: (selector: (snapshot: unknown) => unknown) => selector(useModelSnapshot()),
  useOpenModelInManager: () => undefined,
}));
// The picker's popover and catalog transport are its own tests' concern; a native select keeps the same contract.
vi.mock('@features/models/react', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ModelSelect: ({
    filter,
    id,
    modelTypes,
    onChange,
    placeholder,
    value,
  }: {
    filter?: (candidate: ModelConfig) => boolean;
    id?: string;
    modelTypes: readonly string[];
    onChange: (candidate: ModelConfig | null) => void;
    placeholder: string;
    value: string | null;
  }) => {
    if (placeholder === 'Select a video model…') {
      pickerIds.push(id);
    }

    const options = (useModelSnapshot().models as ModelConfig[]).filter(
      (candidate) => modelTypes.includes(candidate.type) && (!filter || filter(candidate))
    );

    return (
      <select
        // oxlint-disable-next-line jsx-a11y/role-supports-aria-props -- marks the trigger the way ModelSelect's does
        aria-haspopup="listbox"
        aria-label={placeholder}
        id={id}
        value={value ?? ''}
        onChange={(event) => onChange(options.find((candidate) => candidate.key === event.target.value) ?? null)}
      >
        <option value="">{placeholder}</option>
        {options.map((candidate) => (
          <option key={candidate.key} value={candidate.key}>
            {candidate.name}
          </option>
        ))}
      </select>
    );
  },
}));
vi.mock('@features/gallery/mediaSlot', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryMediaSlot: () => null,
}));
vi.mock('@features/generation/components', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  NegativePromptField: () => null,
  PositivePromptField: () => null,
}));
vi.mock('@platform/ui/toaster', () => ({ toaster: { create: notify } }));
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useWorkbenchPreferenceSelector: () => false,
}));
vi.mock('@workbench/image-actions/useFindGalleryItem', () => ({ useFindGalleryItem: () => () => undefined }));
vi.mock('@workbench/useOpenWorkbenchWidget', () => ({ useOpenWorkbenchWidget: () => () => undefined }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(useSyncExternalStore(store.subscribe, store.getSnapshot).activeProject),
  useWorkbenchCommands: () => store.commands,
}));

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const generationAdapter = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: () => undefined },
} as unknown as GenerationUiAdapter;

const SOURCE_VIDEO = {
  endFrame: 72,
  fps: 24,
  height: 704,
  numFrames: 121,
  startFrame: 8,
  video_name: 'harbor.mp4',
  width: 1248,
};
const LAST_FRAME = { height: 704, image_name: 'dusk.png', width: 1248 };

let store: WorkbenchInternalStore;
let host: HTMLDivElement;
let root: Root;

const stored = () => getProjectWidgetValues(store.getSnapshot().activeProject, 'video') as unknown as VideoWidgetValues;
const modelSelect = () => host.querySelector<HTMLSelectElement>('select[aria-label="Select a video model…"]')!;
const dialog = () => document.querySelector<HTMLElement>('[role="alertdialog"]');
const dialogButton = (label: string) =>
  [...(dialog()?.querySelectorAll('button') ?? [])].find((button) => button.textContent === label)!;
const waitForDialogToClose = () => vi.waitFor(() => expect(dialog()).toBeNull());
/** The dialog arms its focus trap a frame after opening; a person cannot answer it sooner. */
const waitForDialogFocus = () => vi.waitFor(() => expect(dialog()?.contains(document.activeElement)).toBe(true));

/** An extend setup on LTX-2 Dev with tuned sampling: an initial video trimmed to frames 8-72, an end frame, a LoRA. */
const render = async () => {
  store.commands.widgets.patchValues('video', {
    ...getDefaultVideoSettings(ltxDev as never, catalog as never),
    lastFrameImage: LAST_FRAME,
    loras: [{ isEnabled: true, model: inkWash, weight: 0.75 }],
    model: ltxDev,
    modelKey: ltxDev.key,
    positivePrompt: 'a harbor at dusk',
    seed: 4242,
    seedMode: 'fixed',
    sourceVideo: SOURCE_VIDEO,
    steps: 37,
  });
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            <GenerationUiProvider adapter={generationAdapter}>
              <VideoUiAdapterProvider>
                <VideoWidgetView />
              </VideoUiAdapterProvider>
            </GenerationUiProvider>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    )
  );
};

const choose = async (key: string) => {
  await act(async () => {
    await userEvent.selectOptions(modelSelect(), key);
  });
};

/** One keyboard step on the LoRA's weight slider: a 250 ms draft, not yet in the store. */
const nudgeLoraWeight = () =>
  act(() => {
    host
      .querySelector('[role="listitem"] [role="slider"]')
      ?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
  });

beforeEach(() => {
  modelStore.set(catalog);
  store = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  notify.mockClear();
  pickerIds.length = 0;
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('video model switch confirmation', () => {
  it('lists what an incompatible model discards, keeps everything on cancel, and clears exactly that on confirm', async () => {
    await render();
    const before = stored();

    await choose(wanT2v.key);

    const confirmedList =
      'Switching to Wan 2.2 T2V removes: the initial video (with its trim), the last frame, and 1 LoRA.';
    expect(dialog()?.textContent).toContain(confirmedList);
    expect(dialog()?.textContent).toContain(
      'It also adjusts Target Resolution and Advanced guidance to fit this model.'
    );
    expect(stored()).toBe(before);
    expect(modelSelect().value).toBe(ltxDev.key);

    await act(async () => {
      await userEvent.click(dialogButton('Cancel'));
    });
    await waitForDialogToClose();

    expect(stored()).toBe(before);
    expect(modelSelect().value).toBe(ltxDev.key);

    await choose(wanT2v.key);
    const confirmingDialog = dialog()!;
    let closingText: string | null | undefined;
    // A browser click can return after the exit ends; inspect the text when the dialog actually starts closing.
    const observer = new MutationObserver(() => {
      if (confirmingDialog.getAttribute('data-state') === 'closed') {
        closingText = confirmingDialog.textContent;
      }
    });
    observer.observe(confirmingDialog, { attributes: true, attributeFilter: ['data-state'] });
    try {
      await act(async () => {
        await userEvent.click(dialogButton('Switch model'));
      });

      // The store has switched, but the closing render still shows exactly what was confirmed.
      expect(stored().modelKey).toBe(wanT2v.key);
      await waitForDialogToClose();
      expect(closingText).toContain(confirmedList);
    } finally {
      observer.disconnect();
    }
    expect(stored()).toMatchObject({
      lastFrameImage: null,
      loras: [],
      model: { key: wanT2v.key },
      modelKey: wanT2v.key,
      positivePrompt: 'a harbor at dusk',
      seed: 4242,
      sourceVideo: null,
      steps: 37,
    });
    expect(modelSelect().value).toBe(wanT2v.key);
    // The confirmation already named the losses; no toast repeats them.
    expect(notify).not.toHaveBeenCalled();
  });

  it('switches at once when nothing user-supplied is lost, reporting only the values it re-fit', async () => {
    await render();

    await choose(ltxDistilled.key);

    expect(dialog()).toBeNull();
    expect(stored()).toMatchObject({
      lastFrameImage: LAST_FRAME,
      loras: [{ model: { key: inkWash.key }, weight: 0.75 }],
      modelKey: ltxDistilled.key,
      sourceVideo: SOURCE_VIDEO,
      steps: 8,
    });
    expect(notify).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({
        description: expect.stringContaining('Steps'),
        title: 'Settings adjusted',
        type: 'info',
      })
    );
  });

  it('applies the switch to the settings as they stand at confirmation', async () => {
    await render();
    await choose(wanT2v.key);

    // Another surface edits the panel while the dialog is open: the end frame goes, a second LoRA arrives.
    await act(() =>
      store.commands.widgets.patchValues('video', {
        lastFrameImage: null,
        loras: [...stored().loras, { isEnabled: true, model: filmGrain, weight: 0.4 }],
        positivePrompt: 'a harbor at night',
      })
    );

    expect(dialog()?.textContent).toContain(
      'Switching to Wan 2.2 T2V removes: the initial video (with its trim) and 2 LoRAs.'
    );

    await act(async () => {
      await userEvent.click(dialogButton('Switch model'));
    });

    expect(stored()).toMatchObject({
      lastFrameImage: null,
      loras: [],
      modelKey: wanT2v.key,
      positivePrompt: 'a harbor at night',
      sourceVideo: null,
    });
  });

  it('closes without applying when the target model is uninstalled while the dialog is open', async () => {
    await render();
    const before = stored();
    await choose(wanT2v.key);
    await waitForDialogFocus();

    await act(() => modelStore.set(catalog.filter((entry) => entry.key !== wanT2v.key)));
    await waitForDialogToClose();

    expect(stored()).toBe(before);
    await vi.waitFor(() => expect(document.activeElement).toBe(modelSelect()));

    // Reinstalling it does not bring back a question the user never answered.
    await act(() => modelStore.set(catalog));

    expect(dialog()).toBeNull();
    expect(stored()).toBe(before);
  });

  it('closes without applying when the project changes while the dialog is open', async () => {
    await render();
    const originalId = store.getSnapshot().activeProject.id;
    const before = stored();
    await choose(wanT2v.key);
    await waitForDialogFocus();

    await act(() => store.commands.projects.create());
    await waitForDialogToClose();

    expect(store.getSnapshot().activeProject.id).not.toBe(originalId);
    expect(stored().modelKey).not.toBe(wanT2v.key);
    const original = store.getSnapshot().projects.find((project) => project.id === originalId)!;
    expect(getProjectWidgetValues(original, 'video')).toBe(before);
  });

  it('keeps the model picker on one id across remounts, so its remembered view and filter survive', async () => {
    await render();
    await act(() => root.unmount());
    root = createRoot(host);
    await render();

    expect(new Set(pickerIds).size).toBe(1);
  });

  it('commits a pending LoRA weight to the current model before judging the switch', async () => {
    await render();
    await nudgeLoraWeight();

    expect(stored().loras[0]?.weight).toBe(0.75);

    await choose(wanT2v.key);

    // Committed on LTX-2 Dev, before the dialog asked about dropping it.
    expect(dialog()?.textContent).toContain('1 LoRA');
    expect(stored()).toMatchObject({ loras: [{ weight: 0.8 }], modelKey: ltxDev.key });

    await act(async () => {
      await userEvent.click(dialogButton('Cancel'));
    });
    await waitForDialogToClose();
    await nudgeLoraWeight();
    await choose(ltxDistilled.key);

    // A lossless switch carries the just-committed weight onto the new model.
    expect(dialog()).toBeNull();
    expect(stored()).toMatchObject({
      loras: [{ model: { key: inkWash.key }, weight: 0.85 }],
      modelKey: ltxDistilled.key,
    });
  });

  it('runs from the keyboard and returns focus to the model control', async () => {
    await render();
    const before = stored();
    await act(() => modelSelect().focus());
    await choose(wanT2v.key);

    await waitForDialogFocus();

    await act(async () => {
      await userEvent.keyboard('{Escape}');
    });
    await waitForDialogToClose();

    expect(stored()).toBe(before);
    await vi.waitFor(() => expect(document.activeElement).toBe(modelSelect()));

    await choose(wanT2v.key);
    await waitForDialogFocus();
    await act(() => dialogButton('Switch model').focus());
    await act(async () => {
      await userEvent.keyboard('{Enter}');
    });
    await waitForDialogToClose();

    expect(stored().modelKey).toBe(wanT2v.key);
    await vi.waitFor(() => expect(document.activeElement).toBe(modelSelect()));
  });
});
