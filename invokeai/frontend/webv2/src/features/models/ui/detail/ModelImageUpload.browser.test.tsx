import { ChakraProvider } from '@chakra-ui/react';
import { type AccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
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

/** The picker's own view is covered by its tests; a pick here hands over an item whose full image is a data URL. */
const PICKED_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
vi.mock('@features/gallery/ui/picker/GalleryPickerPopover', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  GalleryPickerPopover: ({ children, onPick }: { children: ReactNode; onPick: (item: unknown) => void }) => (
    <>
      {children}
      <button
        aria-label="Pick"
        data-gallery-pick
        type="button"
        onClick={() => onPick({ fullUrl: PICKED_PNG, kind: 'image', name: 'picked.png' })}
      />
    </>
  ),
}));

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

  const pick = () => act(() => host.querySelector<HTMLButtonElement>('[data-gallery-pick]')!.click());

  it('sets a gallery image as the cover from its full image, only while its account lifetime is current', async () => {
    await pick();
    await vi.waitFor(() => expect(onUpdated).toHaveBeenCalledOnce());

    const [key, file, signal] = dependencies.updateModelImage.mock.calls[0] as [string, File, AbortSignal];
    expect(key).toBe(model.key);
    expect(file).toBeInstanceOf(File);
    expect(file.name).toBe('picked.png');
    expect(file.type).toBe('image/png');
    expect(signal).toBe(owner.signal);
    expect(dependencies.markCoverImageChanged).toHaveBeenCalledWith(model.key, true);
    expect(onError).not.toHaveBeenCalled();
  });

  it('offers no separate upload: new files arrive through the gallery picker', () => {
    expect(host.querySelector('input[type="file"]')).toBeNull();
    expect([...host.querySelectorAll('button')].map((button) => button.textContent)).not.toContain(
      'widgets.gallery.picker.upload'
    );
  });

  it('quietly drops an account A cover that resolves after account B activates', async () => {
    const request = deferred<void>();
    dependencies.updateModelImage.mockReturnValueOnce(request.promise);

    await pick();
    await vi.waitFor(() => expect(dependencies.updateModelImage).toHaveBeenCalled());

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
