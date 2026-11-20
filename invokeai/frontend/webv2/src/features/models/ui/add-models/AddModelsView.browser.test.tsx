import type { FoundModel } from '@features/models/core/types';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { scanFolderForModels } from '@features/models/data/api';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { type ModelsUiAdapter, ModelsUiProvider, NO_SCHEDULER_POLICY } from '@features/models/ui/ModelsUiContext';
import { getModelsUiSnapshotForTests, setAddModelsQuery, updateModelsUi } from '@features/models/ui/uiStore';
import { createExternalStore } from '@platform/state/externalStore';
import { ApiError } from '@platform/transport/http';
import { system } from '@theme/system';
import { act, StrictMode, useMemo } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { AddModelsView } from './AddModelsView';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: { reason?: string }) =>
      key === 'models.scanFailedDescription' ? `${key}: ${values?.reason ?? ''}` : key,
  }),
}));

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

const DEFAULT_STARTER_RESPONSE = { starter_bundles: {}, starter_models: INSTALLED_STARTERS };
let starterResponse: object = DEFAULT_STARTER_RESPONSE;

vi.mock('@features/models/data/startersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureStartersLoaded: vi.fn(),
  useStartersSelector: (selector: (snapshot: unknown) => unknown) =>
    selector({ error: null, response: starterResponse, status: 'loaded' }),
}));

vi.mock('@features/models/data/externalProvidersStore', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureExternalProvidersLoaded: () => Promise.resolve(),
  useExternalProvidersSelector: (selector: (snapshot: unknown) => unknown) => selector({ configs: [] }),
}));

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

const accountPreferences = createExternalStore({ installInPlace: true });
const setInstallInPlace = (installInPlace: boolean) => accountPreferences.setSnapshot({ installInPlace });

const WithAccountPreferences = ({ children }: { children: ReactNode }) => {
  const { installInPlace } = accountPreferences.useSnapshot();
  const adapter = useMemo<ModelsUiAdapter>(
    () => ({
      canManageModels: true,
      enableModelDescriptions: true,
      installInPlace,
      isProjectActive: () => true,
      managerProjectId: null,
      schedulerPolicy: NO_SCHEDULER_POLICY,
      setInstallInPlace,
    }),
    [installInPlace]
  );

  return <ModelsUiProvider adapter={adapter}>{children}</ModelsUiProvider>;
};

const resetSessionAndPreferences = () => {
  setAddModelsQuery('');
  setInstallInPlace(true);
};

describe('AddModelsView search box', () => {
  let host: HTMLDivElement;
  let root: Root;

  const searchBox = () => document.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]');

  const mount = async () => {
    root = createRoot(host);

    await act(async () => {
      root.render(
        <StrictMode>
          <ChakraProvider value={system}>
            <WithAccountPreferences>
              <AddModelsView />
            </WithAccountPreferences>
          </ChakraProvider>
        </StrictMode>
      );
      // Chakra's observer-driven commits land a task after the render that armed them; awaiting inside this scope keeps
      // them in it.
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };

  const unmount = async () => {
    await act(() => root.unmount());
  };

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);

    resetSessionAndPreferences();
  });

  afterEach(() => {
    host.remove();
    resetSessionAndPreferences();
  });

  it('keeps the typed text and the Install in place choice when the view is reopened', async () => {
    await mount();
    const outside = '/home/user/downloads/outside.safetensors';
    const inplaceInput = () =>
      [...document.querySelectorAll('label')]
        .find((label) => label.textContent === 'models.installInPlace')
        ?.querySelector('input');

    await act(() => userEvent.fill(searchBox()!, outside));
    expect(inplaceInput()?.checked).toBe(true);
    await act(() => userEvent.click(inplaceInput()!.closest('label')!));
    expect(inplaceInput()?.checked).toBe(false);

    await unmount();
    await mount();

    expect(searchBox()?.value).toBe(outside);
    expect(inplaceInput()?.checked).toBe(false);

    await unmount();
  });

  it('sends FP8 storage with a pulled file only while it is ticked', async () => {
    installActions.install.mockReset();
    await mount();

    const source = '/models/qwen_image.safetensors';
    const pull = () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'models.pull');

    await act(() => userEvent.fill(searchBox()!, source));
    await act(() => userEvent.click(pull()!));

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
});

describe('AddModelsView models folder scan', () => {
  let host: HTMLDivElement;
  let root: Root;

  const mount = async () => {
    root = createRoot(host);

    await act(async () => {
      root.render(
        <ChakraProvider value={system}>
          <WithAccountPreferences>
            <AddModelsView />
          </WithAccountPreferences>
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
    resetSessionAndPreferences();
  });

  it('always installs from inside the models folder in place, showing the option on and disabled', async () => {
    const modelsDir = '/opt/invokeai/models';
    setModelsSnapshotForTests({ models: [], modelsDir, status: 'loaded' });
    installActions.install.mockReset();
    installActions.installMany.mockReset();
    await mount();

    const searchBox = document.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]')!;
    const inplaceLabel = () =>
      [...document.querySelectorAll('label')].find((label) => label.textContent === 'models.installInPlace')!;
    const inplaceInput = () => inplaceLabel().querySelector('input')!;

    const outside = '/home/user/downloads/outside.safetensors';
    await act(() => userEvent.fill(searchBox, outside));
    await act(() => userEvent.click(inplaceLabel()));
    expect(inplaceInput().checked).toBe(false);
    await act(() => userEvent.click(buttonNamed('models.pull')!));
    expect(installActions.install).toHaveBeenLastCalledWith(
      expect.objectContaining({ inplace: false, source: outside })
    );

    const inside = `${modelsDir}/loras/inside.safetensors`;
    await act(() => userEvent.fill(searchBox, inside));
    expect(inplaceInput().checked).toBe(true);
    expect(inplaceInput().disabled).toBe(true);
    expect(document.getElementById(inplaceInput().getAttribute('aria-describedby')!)?.textContent).toBe(
      'models.installInPlaceLockedTooltip'
    );
    await act(() => userEvent.click(buttonNamed('models.pull')!));
    expect(installActions.install).toHaveBeenLastCalledWith(expect.objectContaining({ inplace: true, source: inside }));

    expect(accountPreferences.getSnapshot().installInPlace).toBe(false);
    await act(() => userEvent.fill(searchBox, outside));
    expect(inplaceInput().checked).toBe(false);
    expect(inplaceInput().disabled).toBe(false);

    await act(() =>
      updateModelsUi({
        scan: {
          path: '/home/user/invokeai/models',
          results: [
            { is_installed: false, path: `${modelsDir}/main/a.safetensors` },
            { is_installed: false, path: `${modelsDir}/main/b.safetensors` },
          ],
        },
      })
    );
    expect(inplaceInput().checked).toBe(true);
    expect(inplaceInput().disabled).toBe(true);
    await act(() => userEvent.click(buttonNamed('models.installAllCount')!));

    const [requests] = installActions.installMany.mock.lastCall as [{ inplace?: boolean }[]];
    expect(requests.map((request) => request.inplace)).toEqual([true, true]);
  });

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
});

describe('AddModelsView bundle preview decoder', () => {
  let host: HTMLDivElement;
  let root: Root;

  const TAEF1 = {
    base: 'flux',
    compatible_bases: ['z-image'],
    description: 'Preview decoder.',
    is_installed: false,
    name: 'TAEF1 (FLUX.1 preview decoder)',
    source: 'https://example/taef1.safetensors',
    type: 'tiny_autoencoder',
  };
  const TAESDXL = { ...TAEF1, base: 'sdxl', compatible_bases: [], name: 'TAESDXL', source: 'https://example/taesdxl' };
  const FLUX_ONLY_DECODER = { ...TAEF1, compatible_bases: [], name: 'FLUX-only decoder', source: 'https://example/fo' };
  const Z_IMAGE = {
    base: 'z-image',
    description: 'Z-Image main model.',
    is_installed: false,
    name: 'Z-Image Turbo',
    source: 'https://example/z-image-turbo',
    type: 'main',
  };
  const Z_IMAGE_ENCODER = {
    ...Z_IMAGE,
    name: 'Z-Image encoder',
    source: 'https://example/qwen3',
    type: 'qwen3_encoder',
  };

  const useCatalog = (decoderInstalled: boolean) => {
    const decoder = { ...TAEF1, is_installed: decoderInstalled };
    starterResponse = {
      starter_bundles: { 'z-image': { models: [Z_IMAGE, Z_IMAGE_ENCODER], name: 'Z-Image' } },
      starter_models: [...INSTALLED_STARTERS, Z_IMAGE, Z_IMAGE_ENCODER, decoder, TAESDXL, FLUX_ONLY_DECODER],
    };
  };

  const mount = async () => {
    root = createRoot(host);
    await act(async () => {
      root.render(
        <ChakraProvider value={system}>
          <WithAccountPreferences>
            <AddModelsView />
          </WithAccountPreferences>
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
    });
  };
  const selectBundle = () =>
    act(() =>
      userEvent.click(
        [...document.querySelectorAll('button')].find((button) => button.textContent?.startsWith('Z-Image'))!
      )
    );
  const hint = () => document.querySelector<HTMLButtonElement>('button[aria-label="models.showStarterModel"]');
  const listedNames = () => [...document.querySelectorAll('[title]')].map((element) => element.getAttribute('title'));
  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    starterResponse = DEFAULT_STARTER_RESPONSE;
    updateModelsUi({ selectedBundleName: null });
    resetSessionAndPreferences();
  });

  it('points a bundle at its missing decoder, and reveals it in the starters filtered to the family', async () => {
    useCatalog(false);
    updateModelsUi({ addModelsTypeSeed: 'main' });
    setAddModelsQuery('turbo');
    await mount();
    await selectBundle();

    expect(getModelsUiSnapshotForTests().selectedBundleName).toBe('Z-Image');
    expect(listedNames()).not.toContain(TAEF1.name);
    expect(hint()?.textContent).toBe(TAEF1.name);

    hint()!.focus();
    await act(() => userEvent.keyboard('{Enter}'));

    expect(getModelsUiSnapshotForTests().selectedBundleName).toBeNull();
    expect(hint()).toBeNull();
    await expect.poll(listedNames).toContain(TAEF1.name);
    expect(listedNames()).not.toContain(Z_IMAGE.name);
    expect(listedNames()).not.toContain(Z_IMAGE_ENCODER.name);
    expect(listedNames()).not.toContain(INSTALLED_STARTERS[0]!.name);
    expect(listedNames()).not.toContain(TAESDXL.name);
    expect(listedNames()).not.toContain(FLUX_ONLY_DECODER.name);
    expect(getModelsUiSnapshotForTests().addModelsQuery).toBe('');
    await expect
      .poll(() => document.activeElement?.closest('[data-list-row]')?.getAttribute('data-list-row'))
      .toBe(TAEF1.source);
    expect(document.activeElement?.tagName).toBe('BUTTON');
    expect(installActions.installMany).not.toHaveBeenCalled();
  });
});
