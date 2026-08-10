import { ChakraProvider } from '@chakra-ui/react';
import { type AccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReloadNodesButton } from './ReloadNodesButton';

const dependencies = vi.hoisted(() => ({
  error: vi.fn(),
  refreshCustomNodePacks: vi.fn(),
  reloadCustomNodes: vi.fn(),
  success: vi.fn(),
  warning: vi.fn(),
}));

vi.mock('@features/nodes/data/api', () => ({ reloadCustomNodes: dependencies.reloadCustomNodes }));
vi.mock('@features/nodes/data/nodesStore', () => ({
  refreshCustomNodePacks: dependencies.refreshCustomNodePacks,
}));
vi.mock('@features/nodes/ui/useNodesNotify', () => ({
  useNotify: () => ({ error: dependencies.error, success: dependencies.success, warning: dependencies.warning }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const deferred = <Value,>() => {
  let reject!: (reason?: unknown) => void;
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    reject = rejectPromise;
    resolve = resolvePromise;
  });

  return { promise, reject, resolve };
};

describe('ReloadNodesButton account ownership', () => {
  let host: HTMLDivElement;
  let root: Root;
  let owner: AccountScope;

  beforeEach(async () => {
    dependencies.error.mockReset();
    dependencies.refreshCustomNodePacks.mockReset().mockResolvedValue(undefined);
    dependencies.reloadCustomNodes.mockReset().mockResolvedValue({ status: 'Custom nodes reloaded successfully.' });
    dependencies.success.mockReset();
    dependencies.warning.mockReset();
    owner = accountLifecycle.activate('reload-nodes-a', ':user:reload-nodes-a');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ReloadNodesButton />
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('completes reload and refresh within one account lifetime', async () => {
    await act(async () => {
      host.querySelector<HTMLButtonElement>('button')?.click();
      await vi.waitFor(() => expect(dependencies.success).toHaveBeenCalledOnce());
    });

    expect(dependencies.reloadCustomNodes).toHaveBeenCalledWith(owner.signal);
    expect(dependencies.refreshCustomNodePacks).toHaveBeenCalledWith(owner);
    expect(dependencies.error).not.toHaveBeenCalled();
  });

  it('warns instead of celebrating when the backend reports a non-success status', async () => {
    dependencies.reloadCustomNodes.mockResolvedValueOnce({ status: 'No custom nodes directory found.' });

    await act(async () => {
      host.querySelector<HTMLButtonElement>('button')?.click();
      await vi.waitFor(() => expect(dependencies.warning).toHaveBeenCalledOnce());
    });

    expect(dependencies.warning).toHaveBeenCalledWith('nodes.reloadNotice', 'No custom nodes directory found.');
    expect(dependencies.success).not.toHaveBeenCalled();
    expect(dependencies.error).not.toHaveBeenCalled();
  });

  it('keeps an account A abort quiet after account B activates', async () => {
    const request = deferred<{ status: string }>();
    dependencies.reloadCustomNodes.mockReturnValueOnce(request.promise);

    await act(async () => {
      host.querySelector<HTMLButtonElement>('button')?.click();
      await Promise.resolve();
    });
    expect(dependencies.reloadCustomNodes).toHaveBeenCalledWith(owner.signal);

    await act(async () => {
      accountLifecycle.activate('reload-nodes-b', ':user:reload-nodes-b');
      request.reject(new DOMException('The operation was aborted.', 'AbortError'));
      await request.promise.catch(() => undefined);
    });

    expect(owner.signal.aborted).toBe(true);
    expect(dependencies.refreshCustomNodePacks).not.toHaveBeenCalled();
    expect(dependencies.success).not.toHaveBeenCalled();
    expect(dependencies.error).not.toHaveBeenCalled();
  });
});
