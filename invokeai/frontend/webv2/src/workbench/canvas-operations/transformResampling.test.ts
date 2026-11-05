import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { describe, expect, it, vi } from 'vitest';

import { createTransformResampler, type TransformResamplingDeps } from './transformResampling';

const harness = () => {
  const backend = createTestStubRasterBackend();
  const close = vi.fn();
  backend.createImageBitmap = vi.fn(() => Promise.resolve({ width: 3, height: 2, close } as unknown as ImageBitmap));
  const upload = vi.fn<TransformResamplingDeps['upload']>(() => Promise.resolve({ imageName: 'supersampled.png' }));
  const runGraph = vi.fn<TransformResamplingDeps['runGraph']>(() =>
    Promise.resolve({ imageName: 'resized.png', width: 3, height: 2 })
  );
  const resolveImage = vi.fn<TransformResamplingDeps['resolveImage']>(() => Promise.resolve(new Blob(['decoded'])));
  const lifetime = new AbortController();
  const resample = createTransformResampler({ backend, lifetime: lifetime.signal, upload, runGraph, resolveImage });
  return { backend, close, lifetime, resample, resolveImage, runGraph, upload };
};

describe('transform resampling adapter', () => {
  it.each(['bilinear', 'bicubic', 'hamming', 'lanczos'] as const)(
    'runs Pillow %s with requested dimensions and disposes the decoded bitmap',
    async (mode) => {
      const h = harness();
      const output = await h.resample(
        h.backend.createSurface(6, 4),
        { width: 3, height: 2 },
        mode,
        new AbortController().signal
      );
      expect(await h.upload.mock.calls[0]?.[0].text()).toBe('stub-surface-6x4');
      expect(h.runGraph).toHaveBeenCalledWith(
        expect.objectContaining({
          graph: expect.objectContaining({
            nodes: {
              transform_resample: {
                id: 'transform_resample',
                type: 'img_resize',
                is_intermediate: true,
                image: { image_name: 'supersampled.png' },
                width: 3,
                height: 2,
                resample_mode: mode,
              },
            },
          }),
          outputNodeId: 'transform_resample',
          prepend: true,
        })
      );
      expect(h.resolveImage).toHaveBeenCalledWith('resized.png', expect.any(AbortSignal));
      expect(output.width).toBe(3);
      expect(output.height).toBe(2);
      expect(h.close).toHaveBeenCalledOnce();
    }
  );

  it('does not decode an output whose declared dimensions differ', async () => {
    const h = harness();
    h.runGraph.mockResolvedValue({ imageName: 'wrong.png', width: 30, height: 2 });
    await expect(
      h.resample(h.backend.createSurface(6, 4), { width: 3, height: 2 }, 'lanczos', new AbortController().signal)
    ).rejects.toThrow('unexpected dimensions');
    expect(h.resolveImage).not.toHaveBeenCalled();
  });

  it('closes a bitmap whose actual dimensions differ', async () => {
    const h = harness();
    h.backend.createImageBitmap = vi.fn(() =>
      Promise.resolve({ width: 4, height: 2, close: h.close } as unknown as ImageBitmap)
    );
    await expect(
      h.resample(h.backend.createSurface(6, 4), { width: 3, height: 2 }, 'lanczos', new AbortController().signal)
    ).rejects.toThrow('unexpected dimensions');
    expect(h.close).toHaveBeenCalledOnce();
  });

  it('stops after an account or engine lifetime ends during the upload', async () => {
    const h = harness();
    h.upload.mockImplementation(() => {
      h.lifetime.abort();
      return Promise.resolve({ imageName: 'unused.png' });
    });
    await expect(
      h.resample(h.backend.createSurface(6, 4), { width: 3, height: 2 }, 'lanczos', new AbortController().signal)
    ).rejects.toThrow();
    expect(h.runGraph).not.toHaveBeenCalled();
    expect(h.resolveImage).not.toHaveBeenCalled();
  });
});
