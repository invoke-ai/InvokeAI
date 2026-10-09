import type { FoundModel, ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelsDir, listMissingModels, listModels, scanFolderForModels } from '@features/models/data/api';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, updateModelsUi } from '@features/models/ui/uiStore';
import { ApiError } from '@platform/transport/http';
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

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { reason?: string }) =>
      key === 'models.scanFailedDescription' ? `${key}: ${values?.reason ?? ''}` : key,
  }),
}));

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
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModels: vi.fn(),
  scanFolderForModels: vi.fn(),
}));

const installActions = vi.hoisted(() => ({ install: vi.fn(), installMany: vi.fn() }));

vi.mock('./useInstallActions', () => ({
  useInstallActions: () => ({ ...installActions, pendingSources: new Set() }),
}));

const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));

vi.mock('@features/models/ui/useModelsNotify', () => ({ useNotify: () => notify }));

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

describe('AddModelsView models folder scan', () => {
  let host: HTMLDivElement;
  let root: Root;

  const mount = async () => {
    root = createRoot(host);

    await act(async () => {
      root.render(
        <ChakraProvider value={system}>
          <AddModelsView />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };
  const buttonNamed = (label: string) =>
    [...document.querySelectorAll('button')].find((button) => button.textContent === label) ?? null;
  const scanModelsFolder = () => buttonNamed('models.scanModelsFolder');

  /** A scan the test settles by hand; it rejects the way fetch does when its signal aborts. */
  const holdScan = () => {
    let settle!: (results: FoundModel[]) => void;
    let signal!: AbortSignal;

    vi.mocked(scanFolderForModels).mockImplementationOnce(
      (_path, requestSignal) =>
        new Promise<FoundModel[]>((resolve, reject) => {
          settle = resolve;
          signal = requestSignal!;
          signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        })
    );

    return { settle: (results: FoundModel[]) => settle(results), signal: () => signal };
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    vi.mocked(scanFolderForModels).mockReset();
    notify.error.mockClear();
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    updateModelsUi({ hfLookup: null, scan: null });
    setModelsSnapshotForTests({ models: [], modelsDir: null, status: 'loaded' });
  });

  it.each(['/opt/invokeai/models', '/mnt/fast disk/Invoke Models'])(
    'scans the configured models folder %s with one click and shows what it found there',
    async (modelsDir) => {
      setModelsSnapshotForTests({ models: [], modelsDir, status: 'loaded' });
      const scan = holdScan();
      await mount();

      // The resolved path is shown before scanning, and describes the button.
      const pathText = document.getElementById(scanModelsFolder()!.getAttribute('aria-describedby')!);
      expect(pathText?.getAttribute('title')).toBe(modelsDir);

      await act(() => userEvent.click(scanModelsFolder()!));

      expect(scanFolderForModels).toHaveBeenCalledExactlyOnceWith(modelsDir, expect.any(AbortSignal));
      // Progress stays on the row that started it, as Stop.
      expect(scanModelsFolder()).toBeNull();
      expect(buttonNamed('models.stopScan')).not.toBeNull();

      await act(async () => {
        scan.settle([{ is_installed: false, path: `${modelsDir}/main/flux.safetensors` }]);
        await Promise.resolve();
      });

      expect(getModelsUiSnapshotForTests().scan).toEqual({
        path: modelsDir,
        results: [{ is_installed: false, path: `${modelsDir}/main/flux.safetensors` }],
      });
    }
  );

  it('stops a models folder scan without reporting a failure, and can scan again', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
    const scan = holdScan();
    await mount();

    await act(() => userEvent.click(scanModelsFolder()!));
    await act(() => userEvent.click(buttonNamed('models.stopScan')!));

    expect(scan.signal().aborted).toBe(true);
    await vi.waitFor(() => expect(scanModelsFolder()).not.toBeNull());
    expect(scanModelsFolder()).not.toBeDisabled();
    expect(notify.error).not.toHaveBeenCalled();
    expect(getModelsUiSnapshotForTests().scan).toBeNull();
  });

  it('reports a failed scan with the server’s reason, not its raw response, and leaves the action ready to retry', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
    vi.mocked(scanFolderForModels).mockRejectedValueOnce(
      new ApiError(JSON.stringify({ detail: "Permission denied: '/opt/invokeai/models'" }), 500)
    );
    await mount();

    await act(() => userEvent.click(scanModelsFolder()!));

    expect(notify.error).toHaveBeenCalledWith(
      'models.scanFailed',
      "models.scanFailedDescription: Permission denied: '/opt/invokeai/models'"
    );
    await vi.waitFor(() => expect(scanModelsFolder()).not.toBeDisabled());
  });

  it('shows Stop only on the control that started the scan', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
    const scan = holdScan();
    await mount();
    const searchBox = document.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]')!;

    // A folder typed in the field offers its own Scan; the models folder scan owns the one Stop.
    await act(() => userEvent.fill(searchBox, '/data/other-models'));
    await act(() => userEvent.click(scanModelsFolder()!));

    expect(buttonNamed('models.stopScan')).not.toBeNull();
    expect([...document.querySelectorAll('button')].filter((b) => b.textContent === 'models.stopScan')).toHaveLength(1);
    expect(buttonNamed('models.scan')).toBeDisabled();
    expect(scanModelsFolder()).toBeNull();

    await act(async () => {
      scan.settle([]);
      await Promise.resolve();
    });
    await act(() => updateModelsUi({ scan: null }));
    await vi.waitFor(() => expect(buttonNamed('models.scan')).not.toBeDisabled());

    // Started from the field, the field's control turns into Stop and the models folder action waits.
    const fieldScan = holdScan();
    await act(() => userEvent.click(buttonNamed('models.scan')!));

    expect([...document.querySelectorAll('button')].filter((b) => b.textContent === 'models.stopScan')).toHaveLength(1);
    expect(buttonNamed('models.scan')).toBeNull();
    expect(scanModelsFolder()).toBeDisabled();

    await act(async () => {
      fieldScan.settle([]);
      await Promise.resolve();
    });
  });

  it('keeps the models folder scan’s Stop while other results are showing', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
    const scan = holdScan();
    await mount();

    await act(() => userEvent.click(scanModelsFolder()!));
    // Results from elsewhere arrive (a Hugging Face lookup), which would otherwise hide the models folder row.
    await act(() => updateModelsUi({ hfLookup: { repo: 'owner/repo', urls: ['https://example/a.safetensors'] } }));

    expect(buttonNamed('models.stopScan')).not.toBeNull();
    await act(() => userEvent.click(buttonNamed('models.stopScan')!));
    expect(scan.signal().aborted).toBe(true);
  });

  it('says the folder had nothing, naming it, when the scan finds no models', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
    vi.mocked(scanFolderForModels).mockResolvedValueOnce([]);
    await mount();

    await act(() => userEvent.click(scanModelsFolder()!));

    expect(getModelsUiSnapshotForTests().scan).toEqual({ path: '/opt/invokeai/models', results: [] });
    expect(document.body.textContent).toContain('models.noModelFilesFound');
  });

  it('turns a missing models folder into an error with a retry that asks the server again', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: null, status: 'loaded' });
    vi.mocked(listModels).mockResolvedValue([]);
    vi.mocked(listMissingModels).mockResolvedValue([]);
    vi.mocked(getModelsDir).mockResolvedValueOnce('/srv/invoke/models');
    await mount();

    expect(scanModelsFolder()).toBeNull();
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('models.modelsFolderUnavailable');

    await act(() => userEvent.click(buttonNamed('common.retry')!));

    await vi.waitFor(() => expect(scanModelsFolder()).not.toBeNull());
    expect(getModelsDir).toHaveBeenCalledOnce();
    expect(document.querySelector('[role="alert"]')).toBeNull();
    expect(scanFolderForModels).not.toHaveBeenCalled();
  });

  it('waits for the library to locate the folder instead of offering a scan it cannot send', async () => {
    setModelsSnapshotForTests({ models: [], modelsDir: null, status: 'loading' });
    await mount();

    expect(scanModelsFolder()).toBeDisabled();
    expect(document.querySelector('[role="alert"]')).toBeNull();
  });
});
