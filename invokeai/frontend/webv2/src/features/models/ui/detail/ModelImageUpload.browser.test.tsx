import { ChakraProvider } from '@chakra-ui/react';
import { type AccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { ModelImageUpload } from './ModelImageUpload';

const dependencies = vi.hoisted(() => ({
  deleteModelImage: vi.fn(),
  getModelImageUrl: vi.fn(() => '/model-image.png'),
  markCoverImageChanged: vi.fn(),
  updateModelImage: vi.fn(),
}));

vi.mock('@features/models/data/api', () => ({
  deleteModelImage: dependencies.deleteModelImage,
  getModelImageUrl: dependencies.getModelImageUrl,
  updateModelImage: dependencies.updateModelImage,
}));
vi.mock('@features/models/data/modelsStore', () => ({
  markCoverImageChanged: dependencies.markCoverImageChanged,
  useModelsSelector: (selector: (snapshot: { coverImageVersions: Record<string, number> }) => unknown) =>
    selector({ coverImageVersions: {} }),
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

const model = { cover_image: null, key: 'model-a', name: 'Model A' };

describe('ModelImageUpload account ownership', () => {
  let host: HTMLDivElement;
  let onError: Mock<(message: string) => void>;
  let onUpdated: Mock<() => void>;
  let owner: AccountScope;
  let root: Root;

  beforeEach(async () => {
    dependencies.deleteModelImage.mockReset().mockResolvedValue(undefined);
    dependencies.markCoverImageChanged.mockReset();
    dependencies.updateModelImage.mockReset().mockResolvedValue(undefined);
    onError = vi.fn();
    onUpdated = vi.fn();
    owner = accountLifecycle.activate('model-image-a', ':user:model-image-a');
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelImageUpload model={model} onError={onError} onUpdated={onUpdated} />
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  const upload = async (file: File): Promise<void> => {
    const input = host.querySelector<HTMLInputElement>('input[type="file"]')!;

    Object.defineProperty(input, 'files', { configurable: true, value: [file] });
    input.dispatchEvent(new Event('change', { bubbles: true }));
    await Promise.resolve();
  };

  it('commits an upload only while its account lifetime is current', async () => {
    const file = new File(['image'], 'cover.png', { type: 'image/png' });

    await act(() => upload(file));
    await vi.waitFor(() => expect(onUpdated).toHaveBeenCalledOnce());

    expect(dependencies.updateModelImage).toHaveBeenCalledWith(model.key, file, owner.signal);
    expect(dependencies.markCoverImageChanged).toHaveBeenCalledWith(model.key, true);
    expect(onError).not.toHaveBeenCalled();
  });

  it('quietly drops an account A upload that resolves after account B activates', async () => {
    const request = deferred<void>();
    const file = new File(['image'], 'cover.png', { type: 'image/png' });
    dependencies.updateModelImage.mockReturnValueOnce(request.promise);

    await act(() => upload(file));
    expect(dependencies.updateModelImage).toHaveBeenCalledWith(model.key, file, owner.signal);

    await act(async () => {
      accountLifecycle.activate('model-image-b', ':user:model-image-b');
      request.resolve();
      await request.promise;
    });

    expect(owner.signal.aborted).toBe(true);
    expect(dependencies.markCoverImageChanged).not.toHaveBeenCalled();
    expect(onUpdated).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
