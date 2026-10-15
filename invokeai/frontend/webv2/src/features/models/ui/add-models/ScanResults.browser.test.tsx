import type { ModelConfig, ModelInstallJob } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { addInstallJob, replaceInstallJob } from '@features/models/data/installsStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, updateModelsUi } from '@features/models/ui/uiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { ScanResults } from './ScanResults';

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SCANNED_PATH = '/home/user/downloads/juggernaut.safetensors';

// The shape the backend returns for a local install: source is a structured
// LocalModelSource, not the raw string the frontend sent.
const queuedJob: ModelInstallJob = {
  id: 1,
  source: { inplace: true, path: SCANNED_PATH, type: 'local' },
  status: 'waiting',
};

const installedModel = {
  base: 'sdxl',
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash',
  key: 'installed-key',
  name: 'Juggernaut',
  path: SCANNED_PATH,
  source: SCANNED_PATH,
  source_type: 'path',
  type: 'main',
} as ModelConfig;

const noop = () => undefined;
const NO_PENDING: ReadonlySet<string> = new Set<string>();
const SCAN = { path: '/home/user/downloads', results: [{ is_installed: false, path: SCANNED_PATH }] };

const Harness = () => (
  <ChakraProvider value={system}>
    <ScanResults
      fp8Storage={false}
      inplace
      pendingSources={NO_PENDING}
      scan={SCAN}
      onClear={noop}
      onInstall={noop}
      onInstallAll={noop}
      onSetFp8Storage={noop}
      onSetInplace={noop}
    />
  </ChakraProvider>
);

describe('ScanResults install badge lifecycle', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    accountLifecycle.activate('scan-results-test-a', ':user:scan-results-test-a');
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(<Harness />);
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('walks Install -> Installing -> Installed as the job progresses', async () => {
    expect(host.textContent).toContain('models.install');
    expect(host.textContent).not.toContain('models.installing');

    await act(async () => {
      addInstallJob(queuedJob);
      await Promise.resolve();
    });

    expect(host.textContent).toContain('models.installing');

    // The job completes and the library refresh lands the new model.
    await act(async () => {
      replaceInstallJob({ ...queuedJob, status: 'completed' });
      setModelsSnapshotForTests({ models: [installedModel], status: 'loaded' });
      await Promise.resolve();
    });

    expect(host.textContent).not.toContain('models.installing');
    expect(host.textContent).toContain('models.installed');

    // The installed row links to the model it became.
    updateModelsUi({ activeModelKey: null, activeTab: 'add' });
    const viewModel = [...host.querySelectorAll('button')].find((button) => button.textContent === 'models.viewModel');
    expect(viewModel).toBeDefined();
    await act(() => viewModel!.click());
    expect(getModelsUiSnapshotForTests()).toMatchObject({ activeModelKey: 'installed-key', activeTab: 'details' });
  });
});

describe('ScanResults row context menu', () => {
  let host: HTMLDivElement;
  let root: Root;
  const onInstall = vi.fn<(path: string) => void>();

  const menuItem = (value: string) => document.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);
  const openRowMenu = async (value: string) => {
    await act(() => userEvent.click(host.querySelector<HTMLElement>('[data-list-primary]')!, { button: 'right' }));
    await expect.poll(() => menuItem(value)).not.toBeNull();
  };

  beforeEach(async () => {
    accountLifecycle.activate('scan-results-test-b', ':user:scan-results-test-b');
    setModelsSnapshotForTests({ models: [], status: 'loaded' });
    updateModelsUi({ activeModelKey: null, activeTab: 'add', queueExpanded: false });
    onInstall.mockClear();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ScanResults
            fp8Storage={false}
            inplace
            pendingSources={NO_PENDING}
            scan={SCAN}
            onClear={noop}
            onInstall={onInstall}
            onInstallAll={noop}
            onSetFp8Storage={noop}
            onSetInplace={noop}
          />
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('keeps the row out of the tab order beside its Install button', async () => {
    const installButton = [...host.querySelectorAll('button')].find(
      (button) => button.textContent === 'models.install'
    )!;
    installButton.focus();

    await userEvent.tab({ shift: true });

    expect(document.activeElement).not.toBe(host.querySelector('[data-list-primary]'));
  });

  it('animates the menu out on close, still offering the row it opened for', async () => {
    await openRowMenu('install');
    const menu = document.querySelector('[role="menu"]')!;

    const frames = closingFrames(await recordDialogExit(menu, () => act(() => userEvent.keyboard('{Escape}'))));

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain('models.install');
    }
  });

  it('installs the row from its context menu', async () => {
    await openRowMenu('install');

    await act(() => userEvent.click(menuItem('install')!));

    expect(onInstall).toHaveBeenCalledExactlyOnceWith(SCANNED_PATH);
  });

  it('follows an install that starts while the menu is open to the queue', async () => {
    await openRowMenu('install');

    await act(async () => {
      addInstallJob(queuedJob);
      await Promise.resolve();
    });

    await expect.poll(() => menuItem('view-queue')).not.toBeNull();
    expect(menuItem('install')).toBeNull();
    await act(() => userEvent.click(menuItem('view-queue')!));

    expect(getModelsUiSnapshotForTests().queueExpanded).toBe(true);
    expect(onInstall).not.toHaveBeenCalled();
  });

  it('opens the model an installed row became', async () => {
    await act(async () => {
      setModelsSnapshotForTests({ models: [installedModel], status: 'loaded' });
      await Promise.resolve();
    });
    await openRowMenu('view-model');

    await act(() => userEvent.click(menuItem('view-model')!));

    expect(getModelsUiSnapshotForTests()).toMatchObject({ activeModelKey: 'installed-key', activeTab: 'details' });
  });
});
