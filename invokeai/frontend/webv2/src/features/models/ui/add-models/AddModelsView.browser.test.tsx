import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, updateModelsUi } from '@features/models/ui/uiStore';
import { system } from '@theme/system';
import { act, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { AddModelsView } from './AddModelsView';

/**
 * Verify one-shot external search handover into local input state; the seed must not become account-lived
 * filtering.
 */

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

// One starter the backend marks installed by source, one it marks installed by
// name; the library model behind each is what the row must link to.
const INSTALLED_STARTERS = [
  {
    base: 'sdxl',
    description: 'Pulled from its recorded source.',
    is_installed: true,
    name: 'Juggernaut XL',
    source: 'https://example/juggernaut-xl.safetensors',
    type: 'main',
  },
  {
    base: 'sdxl',
    description: 'Renamed after install.',
    is_installed: true,
    name: 'Dreamshaper XL',
    previous_names: ['Dreamshaper'],
    source: 'https://example/dreamshaper-xl.safetensors',
    type: 'main',
  },
];

vi.mock('@features/models/data/startersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureStartersLoaded: vi.fn(),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, response: { starter_bundles: {}, starter_models: INSTALLED_STARTERS }, status: 'loaded' }),
}));

vi.mock('@features/models/data/externalProvidersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureExternalProvidersLoaded: () => Promise.resolve(),
  useExternalProvidersSelector: (selector: (snapshot: unknown) => unknown) => selector({ configs: [] }),
}));

// Spread the original: sibling modules in this view's graph import other
// members of it, and a narrow factory would break their imports outright.
vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getHuggingFaceModels: vi.fn(),
  scanFolderForModels: vi.fn(),
}));

const installActions = vi.hoisted(() => ({ install: vi.fn(), installMany: vi.fn() }));

vi.mock('./useInstallActions', () => ({
  useInstallActions: () => ({ ...installActions, pendingSources: new Set() }),
}));

vi.mock('@features/models/ui/useModelsNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('AddModelsView search seed', () => {
  let host: HTMLDivElement;
  let root: Root;

  const searchBox = () => document.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]');

  const mount = async () => {
    root = createRoot(host);

    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <AddModelsView />
          </ChakraProvider>
        </StrictMode>
      );
      // Chakra's observer-driven commits land a task after the render that
      // armed them; awaiting inside this scope keeps them in it.
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };

  const unmount = async () => {
    await act(() => root.unmount());
  };

  beforeEach(async () => {
    host = document.createElement('div');
    document.body.append(host);

    const store = await import('@features/models/ui/uiStore');
    store.clearAddModelsSeeds();
  });

  afterEach(() => {
    host.remove();
  });

  it('opens searching for a seeded model, then forgets it on the next open', async () => {
    const { requestAddModelsSearch } = await import('@features/models/ui/uiStore');

    requestAddModelsSearch('Wan 2.2 I2V A14B');
    await mount();

    // StrictMode double-invokes the `useState` initializer, so the read has to
    // survive being run twice — this is the case that catches a take-and-clear
    // initializer.
    expect(searchBox()?.value).toBe('Wan 2.2 I2V A14B');

    await unmount();
    await mount();

    // Unmount resets local search; consumed seeds must not reappear on later visits.
    expect(searchBox()?.value).toBe('');

    await unmount();
  });

  it('opens empty when nothing asked it to search', async () => {
    await mount();

    expect(searchBox()?.value).toBe('');

    await unmount();
  });

  it('sends FP8 storage with a pulled file only while it is ticked', async () => {
    installActions.install.mockReset();
    await mount();

    const source = '/models/qwen_image.safetensors';
    const pull = () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'models.pull');

    await act(() => userEvent.fill(searchBox()!, source));
    await act(() => userEvent.click(pull()!));

    // Unticked sends nothing, so identification can still turn FP8 storage on for an FP8 checkpoint.
    expect(installActions.install).toHaveBeenLastCalledWith(
      expect.objectContaining({ config: undefined, inplace: true, source })
    );

    const fp8Storage = [...document.querySelectorAll('label')].find(
      (label) => label.textContent === 'models.installFp8Storage'
    );
    await act(() => userEvent.click(fp8Storage!));
    await act(() => userEvent.click(pull()!));

    expect(installActions.install).toHaveBeenLastCalledWith(
      expect.objectContaining({ config: { default_settings: { fp8_storage: true } }, inplace: true, source })
    );

    await unmount();
  });

  it.each([
    [
      'folder scan results',
      () =>
        updateModelsUi({
          scan: { path: '/models', results: [{ is_installed: false, path: '/models/qwen_image.safetensors' }] },
        }),
    ],
    [
      'a Hugging Face file list',
      () =>
        updateModelsUi({
          hfLookup: {
            repo: 'owner/repo',
            urls: [
              'https://huggingface.co/owner/repo/resolve/main/a.safetensors',
              'https://huggingface.co/owner/repo/resolve/main/b.safetensors',
            ],
          },
        }),
    ],
  ])('sends FP8 storage with every install from %s once it is ticked there', async (_source, openResults) => {
    installActions.installMany.mockReset();
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
    await mount();
    await act(() => openResults());

    // The results replace the hint row, so the option has to be on the results panel itself.
    const fp8Storage = [...document.querySelectorAll('label')].find(
      (label) => label.textContent === 'models.installFp8Storage'
    );
    await act(() => userEvent.click(fp8Storage!));
    const installAll = [...document.querySelectorAll('button')].find(
      (button) => button.textContent === 'models.installAllCount'
    );
    await act(() => userEvent.click(installAll!));

    const [requests] = installActions.installMany.mock.lastCall as [{ config?: unknown }[]];
    expect(requests.length).toBeGreaterThan(0);
    for (const request of requests) {
      expect(request.config).toEqual({ default_settings: { fp8_storage: true } });
    }

    await act(() => updateModelsUi({ hfLookup: null, scan: null }));
    await unmount();
  });

  it('links each installed starter to the library model it became', async () => {
    setModelsSnapshotForTests({
      models: [
        {
          base: 'sdxl',
          key: 'by-source',
          name: 'Juggernaut XL',
          path: 'main/juggernaut-xl.safetensors',
          source: INSTALLED_STARTERS[0]!.source,
          type: 'main',
        },
        {
          base: 'sdxl',
          key: 'by-name',
          name: 'Dreamshaper',
          path: 'main/dreamshaper.safetensors',
          source: 'https://elsewhere/dreamshaper',
          type: 'main',
        },
      ] as ModelConfig[],
      status: 'loaded',
    });
    await mount();

    const links = [...document.querySelectorAll('button')].filter(
      (button) => button.textContent === 'models.viewModel'
    );
    expect(links).toHaveLength(2);
    for (const [index, key] of ['by-source', 'by-name'].entries()) {
      updateModelsUi({ activeModelKey: null, activeTab: 'add' });
      await act(() => links[index]!.click());
      expect(getModelsUiSnapshotForTests()).toMatchObject({ activeModelKey: key, activeTab: 'details' });
    }

    await unmount();
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
  });
});
