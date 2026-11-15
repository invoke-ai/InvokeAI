/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { MainModelConfig } from '@features/generation/core/types';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';
import type * as MenuModule from '@platform/ui/Menu';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ i18n: {}, t: (key: string) => key }) }));
// The items' import fails as a dropped connection would until `fail` clears. Its own file: a page keeps loaded items.
const load = vi.hoisted(() => ({ fail: true }));
vi.mock('@platform/ui/Menu', async (importOriginal) => {
  const original = await importOriginal<typeof MenuModule>();

  return {
    ...original,
    createMenuPartLoader: <Part,>(importPart: () => Promise<Part>) =>
      original.createMenuPartLoader(() =>
        load.fail ? Promise.reject(new TypeError('Failed to fetch dynamically imported module')) : importPart()
      ),
  };
});

import { GenerateHeaderActions } from './GenerateHeaderActions';

seedArchitectureCapabilities();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model: MainModelConfig = {
  base: 'sdxl',
  default_settings: { cfg_scale: 1, steps: 8 },
  key: 'sdxl-turbo',
  name: 'SDXL Turbo',
  type: 'main',
};

const host = document.createElement('div');
const root = createRoot(host);
const unhandled = vi.fn();

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  window.removeEventListener('unhandledrejection', unhandled);
});

it('says the items could not load, leaves the editor standing, and loads them on a later open', async () => {
  window.addEventListener('unhandledrejection', unhandled);
  document.body.append(host);
  const values = createExternalStoreCore<Record<string, unknown>>({ ...getDefaultGenerateSettings(model), model });
  const adapter = {
    generateValues: values,
    models: { catalog: [model], getBaseLabel: () => 'SDXL' },
    notifications: { error: vi.fn(), info: vi.fn(), reportError: vi.fn(), undoable: vi.fn() },
    presets: { includePrompts: true, presets: [], remove: vi.fn(), rename: vi.fn(), save: vi.fn() },
    project: { activeProjectId: 'project-1' },
    settings: { patchGenerateSettings: vi.fn(), readGenerateValues: () => values.getSnapshot() },
  } as unknown as GenerationUiAdapter;
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <GenerationUiProvider adapter={adapter}>
          <GenerateHeaderActions />
        </GenerationUiProvider>
      </ChakraProvider>
    )
  );
  const trigger = page.getByRole('button', { name: 'widgets.generate.resetSettings' });

  // Pointing at the button warms the items; that load failing is no unhandled rejection.
  await userEvent.hover(trigger);
  await userEvent.click(trigger);
  const failed = page.getByRole('menuitem', { name: 'common.menuPartLoadFailed' });
  await expect.element(failed).toHaveAttribute('aria-disabled', 'true');
  expect(page.getByRole('menu').element().getAttribute('aria-busy')).toBe('false');
  expect(trigger.element().isConnected).toBe(true);

  await userEvent.keyboard('{Escape}');
  await expect.element(page.getByRole('menu')).not.toBeInTheDocument();
  load.fail = false;
  await userEvent.click(trigger);

  await expect.element(page.getByRole('menuitem', { name: 'widgets.generate.applyModelDefaults' })).toBeVisible();
  expect(failed.query()).toBeNull();
  expect(unhandled).not.toHaveBeenCalled();
});
