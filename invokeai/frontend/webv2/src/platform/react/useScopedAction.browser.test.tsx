import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { useScopedAction } from './useScopedAction';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const deferred = <T,>() => {
  let reject!: (reason?: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
};

const Harness = ({
  action,
  onError,
  onResult,
}: {
  action: (owner: AccountScope) => Promise<void>;
  onError: (message: string, error: unknown) => void;
  onResult: (result: Promise<boolean>) => void;
}) => {
  const { isBusy, run } = useScopedAction();

  return (
    <button
      data-busy={isBusy}
      type="button"
      onClick={() => {
        onResult(run(action, onError));
      }}
    >
      run
    </button>
  );
};

describe('useScopedAction', () => {
  let host: HTMLDivElement;
  let root: Root;
  let action: Mock<(owner: AccountScope) => Promise<void>>;
  let onError: Mock<(message: string, error: unknown) => void>;
  let onResult: Mock<(result: Promise<boolean>) => void>;

  const mount = async () => {
    await act(() => {
      root.render(<Harness action={action} onError={onError} onResult={onResult} />);
    });
  };

  const button = () => host.querySelector<HTMLButtonElement>('button')!;

  const click = async () => {
    await act(async () => {
      button().click();
      await Promise.resolve();
    });
  };

  const result = (index: number): Promise<boolean> => onResult.mock.calls[index]![0];

  beforeEach(() => {
    accountLifecycle.activate('scoped-action-test-a', ':user:scoped-action-test-a');
    action = vi.fn();
    onError = vi.fn();
    onResult = vi.fn();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('resolves true on success and releases the busy flag', async () => {
    const request = deferred<void>();
    action.mockReturnValue(request.promise);
    await mount();

    await click();
    expect(button().dataset.busy).toBe('true');

    await act(async () => {
      request.resolve();
      await request.promise;
    });

    await expect(result(0)).resolves.toBe(true);
    expect(button().dataset.busy).toBe('false');
    expect(onError).not.toHaveBeenCalled();
  });

  it('resolves false and routes the failure to onError', async () => {
    action.mockRejectedValue(new Error('save exploded'));
    await mount();

    await click();

    await expect(result(0)).resolves.toBe(false);
    expect(onError).toHaveBeenCalledWith('save exploded', expect.any(Error));
    expect(button().dataset.busy).toBe('false');
  });

  it('ignores a second run while the first is in flight', async () => {
    const request = deferred<void>();
    action.mockReturnValue(request.promise);
    await mount();

    await click();
    await click();

    expect(action).toHaveBeenCalledTimes(1);
    await expect(result(1)).resolves.toBe(false);

    await act(async () => {
      request.resolve();
      await request.promise;
    });

    await expect(result(0)).resolves.toBe(true);
    expect(button().dataset.busy).toBe('false');
  });

  it('resolves false without onError when the account changes before success settles', async () => {
    const request = deferred<void>();
    action.mockReturnValue(request.promise);
    await mount();

    await click();
    accountLifecycle.activate('scoped-action-test-b', ':user:scoped-action-test-b');

    await act(async () => {
      request.resolve();
      await request.promise;
    });

    await expect(result(0)).resolves.toBe(false);
    expect(onError).not.toHaveBeenCalled();
  });

  it('does not report failures from a dead account scope', async () => {
    const request = deferred<void>();
    action.mockReturnValue(request.promise);
    await mount();

    await click();
    accountLifecycle.activate('scoped-action-test-b', ':user:scoped-action-test-b');

    await act(async () => {
      request.reject(new Error('failed after switch'));
      await request.promise.catch(() => undefined);
    });

    await expect(result(0)).resolves.toBe(false);
    expect(onError).not.toHaveBeenCalled();
  });
});
