import { afterEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  getImageFullUrl: vi.fn((imageName: string) => `/full/${imageName}`),
  getImageThumbnailUrl: vi.fn((imageName: string) => `/thumbnail/${imageName}`),
  uploadGalleryImage: vi.fn(),
}));

vi.mock('@workbench/gallery/api', () => galleryApi);

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

const stubCropBrowserApis = (onLoadUrl?: (url: string) => void) => {
  const drawImage = vi.fn();

  class TestImage {
    crossOrigin = '';
    naturalHeight = 600;
    naturalWidth = 800;
    onerror: (() => void) | null = null;
    onload: (() => void) | null = null;

    set src(value: string) {
      onLoadUrl?.(value);
      this.onload?.();
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

  return { drawImage, TestImage };
};

afterEach(() => {
  galleryApi.uploadGalleryImage.mockReset();
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
    galleryApi.uploadGalleryImage.mockResolvedValue({ height: 300, imageName: 'new-crop.png', width: 400 });

    const result = await applyReferenceImageCropSelection(image, cropBox);

    expect(galleryApi.uploadGalleryImage).toHaveBeenCalledWith(expect.any(File), 'none', { isIntermediate: true });
    expect(result).toEqual({
      crop: {
        box: { height: 300, width: 400, x: 200, y: 60 },
        image: { height: 300, image_name: 'new-crop.png', width: 400 },
        ratio: null,
      },
      original: image.original,
    });
  });
});
