import type { MainModelConfig } from '@features/generation/core/types';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GeneratePresetsPopover } from './GeneratePresetsPopover';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          generate: {
            presetApplied: 'Applied preset {{name}}',
            presetIncludePrompts: 'Include prompts',
            savePresetAction: 'Save',
            savePresetTitle: 'Save preset',
            switchChanges: { size: 'Size set to {{width}}×{{height}}.' },
          },
        },
      },
    },
  },
});
seedArchitectureCapabilities();

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ADAPTER = {
  generateValues: { getSnapshot: () => ({}), subscribe: () => () => {} },
  models: { catalog: [] },
  notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn(), undoable: vi.fn() },
  presets: {
    includePrompts: true,
    presets: [
      { id: 'preset-1', label: 'Portrait', values: {} },
      { id: 'preset-2', label: 'Landscape', values: {} },
    ],
    remove: vi.fn(),
    rename: vi.fn(),
  },
  project: { activeProjectId: 'project-1' },
  settings: { patchGenerateSettings: vi.fn() },
} as unknown as GenerationUiAdapter;

/** The popover is a dialog too; this selects the modal dialog's content. */
const modalDialog = () => document.querySelector('[data-scope="dialog"][data-part="content"]');
const renameInput = () => modalDialog()?.querySelector<HTMLInputElement>('input[name="renameValue"]');
/** Untranslated in tests, so every row's rename button shares one label; rows keep catalog order. */
const renameButtons = () => document.querySelectorAll<HTMLElement>('[aria-label="widgets.generate.renamePresetNamed"]');

const openRename = async (index: number) => {
  if (renameButtons().length === 0) {
    await act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="widgets.generate.presets"]')!));
    await expect.poll(() => renameButtons().length).toBeGreaterThan(index);
  }

  await act(() => renameButtons()[index]!.click());
  await expect.poll(() => modalDialog()?.getAttribute('data-state')).toBe('open');
};

const renderAdapter = async (adapter: GenerationUiAdapter) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={adapter}>
            <GeneratePresetsPopover />
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await renderAdapter(ADAPTER);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
});

describe('GeneratePresetsPopover', () => {
  const model: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };
  const openPresets = () =>
    act(() => userEvent.click(document.querySelector<HTMLElement>('[aria-label="widgets.generate.presets"]')!));
  const clickPreset = (label: string) =>
    act(() =>
      [...document.querySelectorAll<HTMLButtonElement>('button')].find((b) => b.textContent === label)!.click()
    );

  it.each(['omitted', 'null'] as const)(
    'applies a preset with %s refiner at once and offers an Undo that brings the refiner back',
    async (slot) => {
      const refiner: MainModelConfig = { base: 'sdxl-refiner', key: 'refiner', name: 'Refiner', type: 'main' };
      const olderPreset: Record<string, unknown> = { ...getDefaultGenerateSettings(model), steps: 12 };
      if (slot === 'omitted') {
        delete olderPreset.refinerModel;
      }
      const undoable = vi.fn();
      const values = {
        current: { ...olderPreset, model, refinerModel: refiner, steps: 30 } as Record<string, unknown>,
      };
      const patchGenerateSettings = vi.fn((patch: Record<string, unknown>) => {
        values.current = { ...values.current, ...patch };
      });
      const adapter = {
        ...ADAPTER,
        generateValues: { getSnapshot: () => values.current, subscribe: () => () => {} },
        models: { ...ADAPTER.models, catalog: [model, refiner] },
        presets: { ...ADAPTER.presets, presets: [{ id: 'older', label: 'Older preset', values: olderPreset }] },
        notifications: { ...ADAPTER.notifications, undoable },
        settings: { ...ADAPTER.settings, patchGenerateSettings, readGenerateValues: () => values.current },
      } as unknown as GenerationUiAdapter;
      await renderAdapter(adapter);
      await openPresets();
      await clickPreset('Older preset');

      expect(patchGenerateSettings).toHaveBeenCalledExactlyOnceWith(
        expect.objectContaining({ refinerModel: null, steps: 12 }),
        'project-1'
      );
      expect(undoable).toHaveBeenCalledExactlyOnceWith('Applied preset Older preset', {
        onUndo: expect.any(Function),
        projectId: 'project-1',
        supersedes: 'generate-settings:project-1',
      });

      undoable.mock.calls[0]![1].onUndo();

      expect(patchGenerateSettings).toHaveBeenLastCalledWith({ refinerModel: refiner, steps: 30 }, 'project-1');
    }
  );

  it('keeps settings newer than a preset, and an Undo leaves a template chosen after applying it', async () => {
    const olderPreset: Record<string, unknown> = { ...getDefaultGenerateSettings(model), steps: 12 };
    delete olderPreset.dynamicPromptsEnabled;
    const template = { id: 'cinematic', name: 'Cinematic', negativePrompt: '', positivePrompt: '{prompt}, 35mm' };
    const undoable = vi.fn();
    const values = {
      current: {
        ...getDefaultGenerateSettings(model),
        dynamicPromptsEnabled: false,
        model,
        promptTemplate: template,
        steps: 30,
      } as Record<string, unknown>,
    };
    const patchGenerateSettings = vi.fn((patch: Record<string, unknown>) => {
      values.current = { ...values.current, ...patch };
    });
    const adapter = {
      ...ADAPTER,
      generateValues: { getSnapshot: () => values.current, subscribe: () => () => {} },
      models: { ...ADAPTER.models, catalog: [model] },
      presets: { ...ADAPTER.presets, presets: [{ id: 'older', label: 'Older preset', values: olderPreset }] },
      notifications: { ...ADAPTER.notifications, undoable },
      settings: { ...ADAPTER.settings, patchGenerateSettings, readGenerateValues: () => values.current },
    } as unknown as GenerationUiAdapter;
    await renderAdapter(adapter);
    await openPresets();
    await clickPreset('Older preset');

    expect(values.current).toMatchObject({ dynamicPromptsEnabled: false, promptTemplate: null, steps: 12 });

    const later = { ...template, id: 'noir', name: 'Noir' };
    values.current = { ...values.current, promptTemplate: later };
    undoable.mock.calls[0]![1].onUndo();

    expect(values.current).toMatchObject({ dynamicPromptsEnabled: false, promptTemplate: later, steps: 30 });
  });

  it('names the size a preset moved to when its stored size does not fit the model', async () => {
    // 1028 is off SDXL's 8 px grid, as a preset saved from a hosted model's size can be.
    const preset = { ...getDefaultGenerateSettings(model), height: 1024, width: 1028 };
    // The current size differs from the preset's, so the preset is not shown as active.
    const currentValues = { ...preset, model, width: 1024 };
    const undoable = vi.fn();
    const patchGenerateSettings = vi.fn();
    const adapter = {
      ...ADAPTER,
      generateValues: { getSnapshot: () => currentValues, subscribe: () => () => {} },
      models: { ...ADAPTER.models, catalog: [model] },
      presets: { ...ADAPTER.presets, presets: [{ id: 'off-grid', label: 'Off grid', values: preset }] },
      notifications: { ...ADAPTER.notifications, undoable },
      settings: { ...ADAPTER.settings, patchGenerateSettings },
    } as unknown as GenerationUiAdapter;
    await renderAdapter(adapter);
    await openPresets();
    await clickPreset('Off grid');

    expect(patchGenerateSettings).toHaveBeenCalledWith(expect.objectContaining({ width: 1032 }), 'project-1');
    expect(undoable).toHaveBeenCalledWith(
      'Applied preset Off grid',
      expect.objectContaining({ message: 'Size set to 1032×1024.' })
    );
  });

  it('saves a preset without prompts when asked, remembers the choice, and keeps the prompts it is applied over', async () => {
    const save = vi.fn();
    const setIncludePrompts = vi.fn();
    const patchGenerateSettings = vi.fn();
    const saved = { ...getDefaultGenerateSettings(model), positivePrompt: 'a castle', steps: 12 };
    const values = { current: { ...saved, model } as Record<string, unknown> };
    const adapter = {
      ...ADAPTER,
      generateValues: { getSnapshot: () => values.current, subscribe: () => () => {} },
      models: { ...ADAPTER.models, catalog: [model] },
      presets: { ...ADAPTER.presets, presets: [], save, setIncludePrompts },
      settings: { ...ADAPTER.settings, patchGenerateSettings },
    } as unknown as GenerationUiAdapter;
    await renderAdapter(adapter);
    await openPresets();
    await act(() =>
      userEvent.click([...document.querySelectorAll('button')].find((b) => b.textContent === 'Save preset')!)
    );
    await expect.poll(() => modalDialog()?.getAttribute('data-state')).toBe('open');

    const includePrompts = modalDialog()!.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
    expect(includePrompts.checked).toBe(true);
    await act(() =>
      userEvent.click(modalDialog()!.querySelector<HTMLElement>('[data-scope="checkbox"][data-part="control"]')!)
    );
    expect(includePrompts.checked).toBe(false);
    await act(() => userEvent.type(renameInput()!, 'Settings only{Enter}'));

    expect(save).toHaveBeenCalledOnce();
    const [label, presetValues] = save.mock.calls[0]! as [string, Record<string, unknown>];
    expect(label).toBe('Settings only');
    expect(presetValues).not.toHaveProperty('positivePrompt');
    expect(presetValues).toMatchObject({ steps: 12 });
    expect(setIncludePrompts).toHaveBeenCalledExactlyOnceWith(false);
    // The dialog closes onto the presets trigger, not the page.
    await expect.poll(() => document.activeElement?.getAttribute('aria-label')).toBe('widgets.generate.presets');

    values.current = { ...saved, model, positivePrompt: 'a lighthouse', steps: 30 };
    await renderAdapter({
      ...adapter,
      presets: { ...adapter.presets, presets: [{ id: 'saved', label: 'Settings only', values: presetValues }] },
    } as unknown as GenerationUiAdapter);
    await expect.poll(() => modalDialog()).toBeNull();
    await openPresets();
    await clickPreset('Settings only');

    expect(patchGenerateSettings).toHaveBeenCalledWith(
      expect.objectContaining({ positivePrompt: 'a lighthouse', steps: 12 }),
      'project-1'
    );
  });

  it('animates the rename dialog out instead of unmounting it on close', async () => {
    await openRename(0);
    expect(renameInput()?.value).toBe('Portrait');

    const frames = closingFrames(
      await recordDialogExit(modalDialog()!, () => act(() => userEvent.keyboard('{Escape}')))
    );
    await expect.poll(() => modalDialog()).toBeNull();

    expect(frames).not.toHaveLength(0);
  });

  it('starts a rename reopened during the exit animation from the newly chosen preset', async () => {
    // The reopen has to land while the closed dialog is still animating out; on a loaded machine a real-length exit
    // animation can end before the Escape press even returns, so it is held for the whole test.
    const holdExit = document.createElement('style');
    holdExit.textContent =
      '[data-scope="dialog"][data-part="content"][data-state="closed"] { animation-duration: 3600s !important; }';
    document.head.append(holdExit);
    onTestFinished(() => holdExit.remove());

    await openRename(0);
    await act(() => userEvent.type(renameInput()!, ' draft'));
    await act(() => userEvent.keyboard('{Escape}'));
    // Still animating out: a dialog reused for the next open would keep this draft and the old preset's name.
    expect(modalDialog()?.getAttribute('data-state')).toBe('closed');

    await openRename(1);

    expect(renameInput()?.value).toBe('Landscape');
  });
});
