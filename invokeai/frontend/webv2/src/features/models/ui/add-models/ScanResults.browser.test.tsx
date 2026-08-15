import type { ModelConfig, ModelInstallJob } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { addInstallJob, replaceInstallJob } from '@features/models/data/installsStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
      inplace
      pendingSources={NO_PENDING}
      scan={SCAN}
      onClear={noop}
      onInstall={noop}
      onInstallAll={noop}
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

    // The install POST resolved: the queued job (backend-shaped source) lands
    // in the installs store.
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
  });
});
