import { ChakraProvider } from '@chakra-ui/react';
import { setCustomNodesSnapshotForTests } from '@features/nodes/data/nodesStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AddNodesView } from './AddNodesView';

const api = vi.hoisted(() => ({
  installCustomNodePack: vi.fn(),
  listCustomNodePacks: vi.fn(),
  reloadCustomNodes: vi.fn(),
  uninstallCustomNodePack: vi.fn(),
}));

const notify = vi.hoisted(() => ({
  error: vi.fn(),
  info: vi.fn(),
  success: vi.fn(),
  warning: vi.fn(),
}));

vi.mock('@features/nodes/data/api', () => api);
vi.mock('@features/nodes/ui/useNodesNotify', () => ({ useNotify: () => notify }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const pack = (name: string) => ({ name, nodeCount: 1, nodeTypes: [], path: `/custom_nodes/${name}` });

const setInputValue = (input: HTMLInputElement, value: string): void => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')!.set!;

  setter.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

describe('AddNodesView', () => {
  let host: HTMLDivElement;
  let root: Root;

  const mount = async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <AddNodesView />
        </ChakraProvider>
      );
    });
  };

  const urlInput = () => host.querySelector<HTMLInputElement>('input')!;
  const installButton = () =>
    [...host.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent.includes('nodes.install')
    )!;

  const typeSource = async (value: string) => {
    await act(async () => {
      setInputValue(urlInput(), value);
      await Promise.resolve();
    });
  };

  const submit = async () => {
    await act(async () => {
      installButton().click();
      await Promise.resolve();
    });
  };

  beforeEach(() => {
    api.installCustomNodePack.mockReset();
    notify.error.mockReset();
    notify.success.mockReset();
    notify.warning.mockReset();
    accountLifecycle.activate('add-nodes-test-a', ':user:add-nodes-test-a');
    setCustomNodesSnapshotForTests({
      customNodesPath: '/custom_nodes',
      nodePacks: [pack('existing-pack')],
      status: 'loaded',
    });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('rejects an invalid source before any request is made', async () => {
    await mount();
    await typeSource('https://github.com/owner/..');

    expect(host.textContent).toContain('nodes.invalidSourceName');
    expect(installButton().disabled).toBe(true);

    await act(async () => {
      urlInput().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
      await Promise.resolve();
    });

    expect(api.installCustomNodePack).not.toHaveBeenCalled();
  });

  it('flags an already-installed pack by its derived name', async () => {
    await mount();
    await typeSource('https://github.com/owner/existing-pack.git');

    expect(host.textContent).toContain('nodes.alreadyInstalledError');
    expect(installButton().disabled).toBe(true);
    expect(api.installCustomNodePack).not.toHaveBeenCalled();
  });

  it('toasts success with the imported workflow count and clears the source', async () => {
    api.installCustomNodePack.mockResolvedValue({
      dependency_file: null,
      message: 'ok',
      name: 'new-pack',
      requires_dependencies: false,
      success: true,
      workflows_imported: 2,
    });
    api.listCustomNodePacks.mockResolvedValue({ customNodesPath: '/custom_nodes', nodePacks: [] });
    await mount();
    await typeSource('https://github.com/owner/new-pack.git');
    await submit();

    expect(api.installCustomNodePack).toHaveBeenCalledWith('https://github.com/owner/new-pack.git', expect.anything());
    expect(notify.success).toHaveBeenCalledWith('nodes.installComplete', 'nodes.installCompleteWithWorkflows');
    expect(urlInput().value).toBe('');
  });

  it('toasts a backend-reported failure', async () => {
    api.installCustomNodePack.mockResolvedValue({
      dependency_file: null,
      message: 'clone failed',
      name: 'new-pack',
      requires_dependencies: false,
      success: false,
      workflows_imported: 0,
    });
    await mount();
    await typeSource('https://github.com/owner/new-pack.git');
    await submit();

    expect(notify.error).toHaveBeenCalledWith('nodes.installFailedTitle', 'clone failed');
  });

  it('keeps the typed source across unmount and remount', async () => {
    await mount();
    await typeSource('https://github.com/owner/persisted-pack.git');

    await act(() => root.unmount());
    root = createRoot(host);
    await mount();

    expect(urlInput().value).toBe('https://github.com/owner/persisted-pack.git');
  });
});
