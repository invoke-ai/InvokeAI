/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { MainModelConfig } from '@features/generation/core/types';
import type { GenerationUiAdapter } from '@features/generation/ui/GenerationUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { GenerationUiProvider } from '@features/generation/ui/GenerationUiContext';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateHeaderActions } from './GenerateHeaderActions';

seedArchitectureCapabilities();

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const model: MainModelConfig = { base: 'sdxl', key: 'sdxl', name: 'SDXL', type: 'main' };

describe('GenerateHeaderActions', () => {
  it('resets to the model defaults at once, with an Undo that restores the earlier values', async () => {
    const values = createExternalStoreCore<Record<string, unknown>>({
      ...getDefaultGenerateSettings(model),
      cfgScale: 4,
      model,
      positivePrompt: 'a lighthouse',
      steps: 12,
    });
    const patchGenerateSettings = vi.fn((patch: Record<string, unknown>) => values.patchSnapshot(patch));
    const undoable = vi.fn();
    const adapter = {
      generateValues: values,
      models: { catalog: [model] },
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
          <GenerationUiProvider adapter={adapter}>
            <GenerateHeaderActions />
          </GenerationUiProvider>
        </ChakraProvider>
      )
    );

    await act(() =>
      host!.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.resetAllToModelDefaults"]')!.click()
    );

    expect(values.getSnapshot()).toMatchObject({ cfgScale: 7, positivePrompt: 'a lighthouse', steps: 30 });
    expect(undoable).toHaveBeenCalledExactlyOnceWith('widgets.generate.resetToModelDefaultsDone', {
      onUndo: expect.any(Function),
      projectId: 'project-1',
      supersedes: 'generate-settings:project-1',
    });

    // Steps were changed again since the reset; the Undo keeps that edit and restores the rest.
    values.patchSnapshot({ steps: 50 });
    await act(() => undoable.mock.calls[0]![1].onUndo());

    expect(values.getSnapshot()).toMatchObject({ cfgScale: 4, steps: 50 });
  });
});
