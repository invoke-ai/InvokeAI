import type { ModelConfig } from '@features/models';
import type { VideoUiAdapter } from '@features/video/ui/VideoUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/models/core/architectureCapabilities.testing';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider, NO_DEFAULTS_RULES } from '@features/models/ui/ModelsUiContext';
import { createDefaultVideoWidgetValues } from '@features/video';
import { VideoModelField } from '@features/video/ui/VideoModelField';
import { VideoUiProvider } from '@features/video/ui/VideoUiContext';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t: (key: string) => key }),
}));
// Only navigation out to the manager is outside this picker integration.
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  Link: () => null,
}));

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const checkpoint = {
  base: 'minimax-h3',
  file_size: 1024,
  format: 'checkpoint',
  key: 'unknown-h3',
  name: 'Unknown H3',
  type: 'main',
  variant: 'novel_task',
} as ModelConfig;
const diffusers = { ...checkpoint, key: 'full-h3', name: 'Full H3', format: 'diffusers', variant: 'fl2va' };
const components = { ...diffusers, key: 'components-h3', name: 'Components H3', components_only: true };
const models = [checkpoint, diffusers, components];
const modelsAdapter = {
  canManageModels: false,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => true,
  managerProjectId: null,
  defaultsRules: NO_DEFAULTS_RULES,
  setInstallInPlace: () => undefined,
};

let host: HTMLDivElement;
let root: Root;
let values: ReturnType<typeof createDefaultVideoWidgetValues>;
const patchValues = vi.fn<VideoUiAdapter['patchValues']>((patch) => {
  values = { ...values, ...(typeof patch === 'function' ? patch(values) : patch) };
});
const adapter = { patchValues } as unknown as VideoUiAdapter;
const search = () => document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
const trigger = () => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!;
const optionNames = () => [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent);
const openWithKeyboard = async () => {
  await act(() => trigger().focus());
  await userEvent.keyboard('{Enter}');
  await expect.poll(() => document.activeElement).toBe(search());
};

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  setModelsSnapshotForTests({ error: null, models, status: 'loaded' });
  values = createDefaultVideoWidgetValues(models);
  patchValues.mockClear();
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <ModelsUiProvider adapter={modelsAdapter}>
          <VideoUiProvider adapter={adapter}>
            <VideoModelField models={models} modelsLoaded projectId="picker-test" values={values} />
          </VideoUiProvider>
        </ModelsUiProvider>
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
});

describe('H3 task header through the real model picker', () => {
  it('starts in search, navigates filtered options, and restores focus on Escape without selecting', async () => {
    await openWithKeyboard();
    expect(optionNames().join(' ')).toContain('Full H3');
    expect(optionNames().join(' ')).not.toContain('Components H3');
    await userEvent.keyboard('{ArrowUp}');
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('Full H3');
    await userEvent.keyboard('{ArrowDown}');
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('Unknown H3');
    await userEvent.keyboard('Unknown');
    await expect.poll(optionNames).toHaveLength(1);
    await userEvent.keyboard('{ArrowDown}');
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('Unknown H3');
    expect(search().getAttribute('aria-activedescendant')).toBe(document.querySelector('[data-active]')?.id);
    await userEvent.keyboard('{Escape}');
    await expect.poll(() => trigger().getAttribute('aria-expanded')).toBe('false');
    await expect.poll(() => document.activeElement).toBe(trigger());
    expect(patchValues).not.toHaveBeenCalled();
  });

  it.each(['Ref2VA', 'FL2VA'] as const)(
    'selects %s explicitly using header and search keyboard navigation',
    async (task) => {
      await openWithKeyboard();
      // Header remains in natural tab order before search, without taking initial focus.
      await userEvent.keyboard('{Shift>}{Tab}{/Shift}');
      await expect.element(page.getByRole('button', { name: 'FL2VA', exact: true })).toHaveFocus();
      if (task === 'Ref2VA') {
        await userEvent.keyboard('{Shift>}{Tab}{/Shift}');
      }
      await userEvent.keyboard('{Enter}');
      await expect
        .element(page.getByRole('button', { name: task, exact: true }))
        .toHaveAttribute('aria-pressed', 'true');
      await expect.poll(optionNames).toHaveLength(1);
      expect(optionNames().join(' ')).toContain('Unknown H3');
      await userEvent.keyboard(task === 'Ref2VA' ? '{Tab}{Tab}' : '{Tab}');
      await expect.poll(() => document.activeElement).toBe(search());
      await userEvent.keyboard('Unknown{ArrowDown}{Enter}');
      await expect.poll(() => trigger().getAttribute('aria-expanded')).toBe('false');
      expect(values.h3TaskOverride).toBe(task === 'Ref2VA' ? 'ref2va' : 'fl2va');
      expect(values.model).toBe(checkpoint);
      expect(values.model?.variant).toBe('novel_task');
      await expect.poll(() => document.activeElement).toBe(trigger());
    }
  );

  it('discards a pending task on Escape and reopens in Detected with an empty search', async () => {
    await openWithKeyboard();
    await page.getByRole('button', { name: 'Ref2VA', exact: true }).click();
    await userEvent.keyboard('{Tab}{Tab}Unknown');
    expect(search().value).toBe('Unknown');
    await userEvent.keyboard('{Escape}');
    await expect.poll(() => document.activeElement).toBe(trigger());
    expect(patchValues).not.toHaveBeenCalled();
    // Wait for the exit animation to unmount the picker before starting a fresh search session.
    await expect.poll(() => document.querySelector('input[role="combobox"]')).toBeNull();
    await openWithKeyboard();
    await expect
      .element(page.getByRole('button', { name: 'widgets.video.h3Detected', exact: true }))
      .toHaveAttribute('aria-pressed', 'true');
    expect(search().value).toBe('');
    await expect.poll(optionNames).toHaveLength(2);
    await userEvent.keyboard('Unknown{ArrowDown}{Enter}');
    expect(values.h3TaskOverride).toBeUndefined();
    expect(values.model).toBe(checkpoint);
  });
});
