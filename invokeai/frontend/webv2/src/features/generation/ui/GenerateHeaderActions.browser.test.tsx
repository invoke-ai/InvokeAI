/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { MainModelConfig } from '@features/generation/core/types';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import type * as MenuModule from '@platform/ui/Menu';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { GenerateHeaderActions } from './GenerateHeaderActions';

// Each test starts with the items unloaded, as a page does: the loaders the module creates are swapped for fresh ones.
const menuParts = vi.hoisted(() => ({
  loaders: [] as { getSnapshot: () => { status: string }; reset: () => void }[],
}));
vi.mock('@platform/ui/Menu', async (importOriginal) => {
  const original = await importOriginal<typeof MenuModule>();

  return {
    ...original,
    createMenuPartLoader: <Part,>(importPart: () => Promise<Part>) => {
      let current = original.createMenuPartLoader(importPart);
      const loader = {
        getSnapshot: () => current.getSnapshot(),
        load: () => current.load(),
        reset: () => {
          current = original.createMenuPartLoader(importPart);
        },
        subscribe: (listener: () => void) => current.subscribe(listener),
      };
      menuParts.loaders.push(loader);
      return loader;
    },
  };
});

seedArchitectureCapabilities();

// The real catalog, so the tooltip and toast read as users see them.
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => menuParts.loaders.forEach((loader) => loader.reset()));

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** A turbo SDXL model that sets steps and CFG in the Model Manager, and no size. */
const turbo: MainModelConfig = {
  base: 'sdxl',
  default_settings: { cfg_scale: 1, steps: 8 },
  key: 'sdxl-turbo',
  name: 'SDXL Turbo',
  type: 'main',
};

const render = async (model: MainModelConfig) => {
  const values = createExternalStoreCore<Record<string, unknown>>({
    ...getDefaultGenerateSettings(model),
    aspectRatioId: 'Free',
    aspectRatioValue: 1216 / 832,
    cfgScale: 7,
    height: 832,
    model,
    positivePrompt: 'a lighthouse',
    steps: 30,
    width: 1216,
  });
  const patchGenerateSettings = vi.fn((patch: Record<string, unknown>) => values.patchSnapshot(patch));
  const undoable = vi.fn();
  const adapter = {
    generateValues: values,
    models: { catalog: [model], getBaseLabel: () => 'SDXL' },
    notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn(), undoable },
    presets: { includePrompts: true, presets: [], remove: vi.fn(), rename: vi.fn(), save: vi.fn() },
    project: { activeProjectId: 'project-1' },
    settings: { patchGenerateSettings, readGenerateValues: () => values.getSnapshot() },
  } as unknown as GenerationUiAdapter;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerationUiProvider adapter={adapter}>
            <GenerateHeaderActions />
          </GenerationUiProvider>
        </I18nextProvider>
      </ChakraProvider>
    )
  );

  return { patchGenerateSettings, undoable, values };
};

const trigger = () => page.getByRole('button', { name: 'Reset settings' });
const item = (name: string) => page.getByRole('menuitem', { name });
const openMenu = async () => {
  await userEvent.click(trigger());
  await expect.element(page.getByRole('menu')).toBeVisible();
};

/** Waits for the warm-up that pointing at or focusing the button starts, without loading the items itself. */
const settleWarmUp = () =>
  vi.waitFor(() => expect(menuParts.loaders.map((loader) => loader.getSnapshot().status)).toEqual(['loaded']));

/** Records whether the placeholder ever mounts while `open` runs. */
const placeholdersDuring = async (open: () => Promise<unknown>): Promise<Node[]> => {
  const placeholders: Node[] = [];
  const observer = new MutationObserver(() =>
    placeholders.push(...document.querySelectorAll('[data-value="menu-part-placeholder"]'))
  );
  observer.observe(document.body, { childList: true, subtree: true });
  await open();
  observer.disconnect();

  return placeholders;
};

describe('GenerateHeaderActions', () => {
  it("opens on its items' first frame once pointing at the button has loaded them", async () => {
    await render(turbo);
    await userEvent.hover(trigger());
    await settleWarmUp();

    const placeholders = await placeholdersDuring(async () => {
      await userEvent.click(trigger());
      await expect.element(item('Apply model defaults')).toBeVisible();
    });

    expect(placeholders).toEqual([]);
    expect(page.getByRole('menu').element().getAttribute('aria-busy')).toBe('false');
  });

  it("opens from the keyboard on its items' first frame once focusing the button has loaded them", async () => {
    await render(turbo);
    await act(() => (trigger().element() as HTMLButtonElement).focus());
    await settleWarmUp();

    const placeholders = await placeholdersDuring(async () => {
      await userEvent.keyboard('{Enter}');
      await expect.element(item('Apply model defaults')).toBeVisible();
    });

    expect(placeholders).toEqual([]);
    expect(page.getByRole('menu').element().getAttribute('aria-busy')).toBe('false');
  });

  it("says what applying changes, changes only the model's own values, and undoes them in one step", async () => {
    const { patchGenerateSettings, undoable, values } = await render(turbo);

    await openMenu();
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription(
        'Steps 30 → 8, CFG 7 → 1. Left as is, not set for this model: Size, Scheduler, VAE, and VAE precision'
      );
    await userEvent.click(item('Apply model defaults'));

    expect(patchGenerateSettings).toHaveBeenCalledExactlyOnceWith({ cfgScale: 1, steps: 8 }, 'project-1');
    expect(values.getSnapshot()).toMatchObject({
      cfgScale: 1,
      height: 832,
      positivePrompt: 'a lighthouse',
      steps: 8,
      width: 1216,
    });
    expect(undoable).toHaveBeenCalledExactlyOnceWith('Applied SDXL Turbo defaults', {
      message: 'Steps 30 → 8, CFG 7 → 1',
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-settings:project-1',
    });

    // Nothing is left to apply, and the item says why.
    await openMenu();
    await expect.element(item('Apply model defaults')).toHaveAttribute('aria-disabled', 'true');
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription(expect.stringContaining("Already using SDXL Turbo's defaults."));
    await userEvent.keyboard('{Escape}');

    // Steps were changed again since; the Undo keeps that edit and restores the rest.
    values.patchSnapshot({ steps: 50 });
    await act(() => undoable.mock.calls[0]![1].onUndo());

    expect(values.getSnapshot()).toMatchObject({ cfgScale: 7, steps: 50 });
  });

  it('resets everything, size included, from the keyboard, returning focus to the button, undone in one step', async () => {
    const { undoable, values } = await render(turbo);

    await act(() => (trigger().element() as HTMLButtonElement).focus());
    await settleWarmUp();
    await userEvent.keyboard('{Enter}');
    await expect
      .element(item('Reset everything to defaults'))
      .toHaveAccessibleDescription('Size 1216×832 → 1024×1024, Aspect ratio Free → 1:1, Steps 30 → 8, CFG 7 → 1');
    // The menu takes focus once open and tracks the highlighted item, the first, by attribute.
    await vi.waitFor(() => expect(document.activeElement?.getAttribute('role')).toBe('menu'));
    await userEvent.keyboard('{ArrowDown}');
    // Zag focuses the menu a frame after each move; Enter in the same frame would race that focus.
    await expect.element(item('Reset everything to defaults')).toHaveAttribute('data-highlighted');
    await new Promise(requestAnimationFrame);
    await userEvent.keyboard('{Enter}');

    expect(values.getSnapshot()).toMatchObject({ cfgScale: 1, height: 1024, steps: 8, width: 1024 });
    expect(undoable).toHaveBeenCalledExactlyOnceWith('Reset everything to defaults', {
      message: 'Size 1216×832 → 1024×1024, Aspect ratio Free → 1:1, Steps 30 → 8, CFG 7 → 1',
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-settings:project-1',
    });
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger().element()));

    await act(() => undoable.mock.calls[0]![1].onUndo());

    expect(values.getSnapshot()).toMatchObject({ cfgScale: 7, height: 832, steps: 30, width: 1216 });
  });

  it('names a ratio lock the full reset sets when the size already matches, in the hint and the toast', async () => {
    const { undoable, values } = await render(turbo);
    values.patchSnapshot({ aspectRatioValue: 1, cfgScale: 1, height: 1024, steps: 8, width: 1024 });

    await openMenu();
    await expect.element(item('Reset everything to defaults')).toHaveAccessibleDescription('Aspect ratio Free → 1:1');
    await userEvent.click(item('Reset everything to defaults'));

    expect(values.getSnapshot()).toMatchObject({ aspectRatioId: '1:1', height: 1024, width: 1024 });
    expect(undoable).toHaveBeenCalledExactlyOnceWith(
      'Reset everything to defaults',
      expect.objectContaining({ message: 'Aspect ratio Free → 1:1' })
    );
  });

  it('tells a default VAE that is not installed apart from what the model does not set', async () => {
    await render({ ...turbo, default_settings: { steps: 8, vae: 'not-installed' } });

    await openMenu();
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription(
        'Steps 30 → 8. Left as is, not set for this model: Size, CFG, Scheduler, and VAE precision. ' +
          "The model's VAE is not installed or does not fit it, so your VAE stays."
      );
  });

  it('says why there is nothing to apply when the only default is a VAE that is not installed', async () => {
    await render({ ...turbo, default_settings: { vae: 'not-installed' }, name: 'VAE only' });

    await openMenu();
    await expect.element(item('Apply model defaults')).toHaveAttribute('aria-disabled', 'true');
    // Told once, in the reason.
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription(
        'Left as is, not set for this model: Size, Steps, CFG, Scheduler, and VAE precision ' +
          'VAE only sets only a VAE, which is not installed or does not fit it, so your VAE stays.'
      );
  });

  it("agrees that the model's values hold except a VAE that is not installed", async () => {
    await render({ ...turbo, default_settings: { steps: 30, vae: 'not-installed' }, name: 'Steps only' });

    await openMenu();
    await expect.element(item('Apply model defaults')).toHaveAttribute('aria-disabled', 'true');
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription(
        'Left as is, not set for this model: Size, CFG, Scheduler, and VAE precision ' +
          "Already using Steps only's defaults, except its VAE, which is not installed or does not fit it, so your " +
          'VAE stays.'
      );
  });

  it('offers no apply, saying why, for a model with no defaults of its own', async () => {
    const { patchGenerateSettings } = await render({ ...turbo, default_settings: null, name: 'Plain SDXL' });

    await openMenu();
    await expect.element(item('Apply model defaults')).toHaveAttribute('aria-disabled', 'true');
    await expect
      .element(item('Apply model defaults'))
      .toHaveAccessibleDescription('Plain SDXL has no default settings of its own. Add them in the Model Manager.');
    // Playwright refuses to click an aria-disabled control, so the click goes to the element itself.
    await act(() => (item('Apply model defaults').element() as HTMLElement).click());

    expect(patchGenerateSettings).not.toHaveBeenCalled();
  });
});
