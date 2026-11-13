import type { UtilityGraphExecutionStatus } from '@features/queue/utility';
import type { ImageResolver, TransformOptions } from '@workbench/canvas-engine/api';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { BackendGraphContract } from '@workbench/graphContracts';

export interface TransformResamplingDeps {
  lifetime: AbortSignal;
  backend: RasterBackend;
  resolveImage: ImageResolver;
  upload(blob: Blob, signal: AbortSignal): Promise<{ imageName: string }>;
  runGraph(options: {
    onStatus?: (status: UtilityGraphExecutionStatus) => void;
    prepend: boolean;
    graph: BackendGraphContract;
    outputNodeId: string;
    signal: AbortSignal;
  }): Promise<{ imageName: string; width: number; height: number }>;
}

/** The v6 transform pipeline: 2x canvas raster followed by Pillow's selected downsampling kernel. */
export const createTransformResampler =
  (deps: TransformResamplingDeps) =>
  async (
    surface: RasterSurface,
    target: { width: number; height: number },
    mode: TransformOptions['resampleMode'],
    signal: AbortSignal,
    onProgress?: (phase: 'queued' | 'resampling') => void
  ): Promise<ImageData> => {
    signal = AbortSignal.any([signal, deps.lifetime]);
    signal.throwIfAborted();
    const blob = await deps.backend.encodeSurface(surface);
    signal.throwIfAborted();
    const image = await deps.upload(blob, signal);
    signal.throwIfAborted();
    const id = 'transform_resample';
    let resamplingObserved = false;
    const output = await deps.runGraph({
      graph: {
        id,
        edges: [],
        nodes: {
          [id]: {
            id,
            type: 'img_resize',
            is_intermediate: true,
            image: { image_name: image.imageName },
            width: target.width,
            height: target.height,
            resample_mode: mode,
          },
        },
      },
      outputNodeId: id,
      prepend: true,
      onStatus: onProgress
        ? (status) => {
            if (!signal.aborted) {
              if (status === 'in_progress') {
                resamplingObserved = true;
              }
              onProgress(status === 'pending' ? 'queued' : 'resampling');
            }
          }
        : undefined,
      signal,
    });
    signal.throwIfAborted();
    if (output.width !== target.width || output.height !== target.height) {
      throw new Error('Transform resize returned unexpected dimensions');
    }
    if (!resamplingObserved) {
      onProgress?.('resampling');
    }
    const result = await deps.resolveImage(output.imageName, signal);
    signal.throwIfAborted();
    const bitmap = await deps.backend.createImageBitmap(result);
    try {
      signal.throwIfAborted();
      if (bitmap.width !== target.width || bitmap.height !== target.height) {
        throw new Error('Transform image decoded at unexpected dimensions');
      }
      const decoded = deps.backend.createSurface(target.width, target.height);
      decoded.ctx.drawImage(bitmap, 0, 0);
      return decoded.ctx.getImageData(0, 0, target.width, target.height);
    } finally {
      bitmap.close();
    }
  };
