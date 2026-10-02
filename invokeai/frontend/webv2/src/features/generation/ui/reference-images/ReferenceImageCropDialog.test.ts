import { type AccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryTransfers: { upload: vi.fn() },
}));
const galleryUtility = vi.hoisted(() => ({
  galleryImageUrls: {
    full: vi.fn((imageName: string) => `/full/${imageName}`),
    thumbnail: vi.fn((imageName: string) => `/thumbnail/${imageName}`),
  },
}));

vi.mock('@features/gallery', () => galleryApi);
vi.mock('@features/gallery/utility', () => galleryUtility);

import { applyReferenceImageCropSelection, exportCroppedReferenceImage } from './ReferenceImageCropDialog';

const image = {
  crop: {
    box: { height: 200, width: 300, x: 10, y: 20 },
    image: { height: 200, image_name: 'previous-crop.png', width: 300 },
    ratio: null,
  },
  original: { image: { height: 600, image_name: 'original.png', width: 800 } },
};
const cropBox = { height: 50, width: 50, x: 25, y: 10 };

const stubCropBrowserApis = (onLoadUrl?: (url: string) => void, autoLoad = true) => {
  const drawImage = vi.fn();
  let completePendingLoad: (() => void) | null = null;

  class TestImage {
    crossOrigin = '';
    naturalHeight = 600;
    naturalWidth = 800;
    onerror: (() => void) | null = null;
    onload: (() => void) | null = null;

    set src(value: string) {
      onLoadUrl?.(value);
      if (autoLoad) {
        this.onload?.();
      } else {
        const onload = this.onload;
        completePendingLoad = () => onload?.();
      }
    }
  }

  vi.stubGlobal('Image', TestImage);
  vi.stubGlobal('document', {
    createElement: () => ({
      getContext: () => ({ drawImage }),
      height: 0,
      toBlob: (callback: (blob: Blob) => void) => callback(new Blob(['crop'], { type: 'image/png' })),
      width: 0,
    }),
  });

  return { completeLoad: () => completePendingLoad?.(), drawImage, TestImage };
};

let owner: AccountScope;

beforeEach(() => {
  owner = accountLifecycle.activate('reference-crop-a', ':user:reference-crop-a');
});

afterEach(() => {
  galleryApi.galleryTransfers.upload.mockReset();
  accountLifecycle.invalidate();
  vi.unstubAllGlobals();
});

describe('ReferenceImageCropDialog', () => {
  it('recrops from the original image rather than the previous crop', async () => {
    let loadedUrl = '';
    const { drawImage, TestImage } = stubCropBrowserApis((url) => {
      loadedUrl = url;
    });

    const file = await exportCroppedReferenceImage(image, cropBox);

    expect(loadedUrl).toBe('/full/original.png');
    expect(loadedUrl).not.toContain('previous-crop.png');
    expect(drawImage).toHaveBeenCalledWith(expect.any(TestImage), 200, 60, 400, 300, 0, 0, 400, 300);
    expect(file.name).toBe('original-crop.png');
  });

  it('uploads an applied crop as intermediate and preserves the original', async () => {
    stubCropBrowserApis();
    galleryApi.galleryTransfers.upload.mockResolvedValue({ height: 300, imageName: 'new-crop.png', width: 400 });

    const result = await applyReferenceImageCropSelection(image, cropBox, owner);

    expect(galleryApi.galleryTransfers.upload).toHaveBeenCalledWith(expect.any(File), 'none', {
      isIntermediate: true,
      signal: owner.signal,
    });
    expect(result).toEqual({
      crop: {
        box: { height: 300, width: 400, x: 200, y: 60 },
        image: { height: 300, image_name: 'new-crop.png', width: 400 },
        ratio: null,
      },
      original: image.original,
    });
  });

  it('does not upload when account A expires during local crop preprocessing', async () => {
    const { completeLoad } = stubCropBrowserApis(undefined, false);
    const pendingCrop = applyReferenceImageCropSelection(image, cropBox, owner);

    accountLifecycle.activate('reference-crop-b', ':user:reference-crop-b');
    completeLoad();

    await expect(pendingCrop).rejects.toHaveProperty('name', 'AccountScopeExpiredError');
    expect(owner.signal.aborted).toBe(true);
    expect(galleryApi.galleryTransfers.upload).not.toHaveBeenCalled();
  });

  it('rejects an account A upload that resolves after account B activates', async () => {
    stubCropBrowserApis();
    let resolveUpload!: (value: { height: number; imageName: string; width: number }) => void;
    const upload = new Promise<{ height: number; imageName: string; width: number }>((resolve) => {
      resolveUpload = resolve;
    });
    galleryApi.galleryTransfers.upload.mockReturnValue(upload);

    const pendingCrop = applyReferenceImageCropSelection(image, cropBox, owner);
    await vi.waitFor(() => expect(galleryApi.galleryTransfers.upload).toHaveBeenCalledOnce());

    accountLifecycle.activate('reference-crop-b', ':user:reference-crop-b');
    resolveUpload({ height: 300, imageName: 'new-crop.png', width: 400 });

    await expect(pendingCrop).rejects.toHaveProperty('name', 'AccountScopeExpiredError');
    expect(owner.signal.aborted).toBe(true);
  });
});
