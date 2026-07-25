import type { CanvasLayerCapability, CanvasPreviewCapability } from '@workbench/canvas-engine/capabilities';

import type { MaskLayerControllerOptions } from './maskLayerController';
import type { StructuralLayerController } from './structuralLayerController';

import { BooleanMergeController, type BooleanMergeControllerOptions } from './booleanMergeController';
import { CopyLayerController, type CopyLayerControllerOptions } from './copyLayerController';
import { CropLayerController, type CropLayerControllerOptions } from './cropLayerController';
import { ExtractMaskedAreaController, type ExtractMaskedAreaControllerOptions } from './extractMaskedAreaController';
import { MaskLayerController } from './maskLayerController';
import { MergeLayerController, type MergeLayerControllerOptions } from './mergeLayerController';
import { NewRasterLayerController, type NewRasterLayerControllerOptions } from './newRasterLayerController';
import { RasterizeLayerController, type RasterizeLayerControllerOptions } from './rasterizeLayerController';
import { ThumbnailController, type ThumbnailControllerOptions } from './thumbnailController';

export type LayerControllerDeps<Permit> = Omit<
  CanvasLayerCapability,
  'applyStructuralPreview' | 'canCommitStructural' | 'commitStagedImage' | 'commitStructural' | 'invertMask'
> &
  Omit<CanvasPreviewCapability, 'drawLayerThumbnail' | 'requestLayerThumbnail'> & {
    mask: MaskLayerControllerOptions;
    thumbnail: ThumbnailControllerOptions;
    structural: StructuralLayerController;
    rasterize: RasterizeLayerControllerOptions;
    merge: MergeLayerControllerOptions;
    booleanMerge: BooleanMergeControllerOptions<Permit>;
    extractMaskedArea: ExtractMaskedAreaControllerOptions<Permit>;
    crop: CropLayerControllerOptions<Permit>;
    copy: CopyLayerControllerOptions<Permit>;
    newRasterLayer: NewRasterLayerControllerOptions<Permit>;
  };

/** Public layer-operation boundary. Implementations are injected by the composition root. */
export class LayerController<Permit> {
  readonly layers: Omit<CanvasLayerCapability, 'commitStagedImage'>;
  readonly previews: CanvasPreviewCapability;
  readonly mask: MaskLayerController;
  readonly thumbnail: ThumbnailController;
  readonly structural: StructuralLayerController;
  readonly rasterize: RasterizeLayerController;
  readonly merge: MergeLayerController;
  readonly booleanMerge: BooleanMergeController<Permit>;
  readonly extractMaskedArea: ExtractMaskedAreaController<Permit>;
  readonly crop: CropLayerController<Permit>;
  readonly copy: CopyLayerController<Permit>;
  readonly newRasterLayer: NewRasterLayerController<Permit>;
  private disposed = false;

  constructor(deps: LayerControllerDeps<Permit>) {
    this.mask = new MaskLayerController(deps.mask);
    this.thumbnail = new ThumbnailController(deps.thumbnail);
    this.structural = deps.structural;
    this.rasterize = new RasterizeLayerController(deps.rasterize);
    this.merge = new MergeLayerController(deps.merge);
    this.booleanMerge = new BooleanMergeController(deps.booleanMerge);
    this.extractMaskedArea = new ExtractMaskedAreaController(deps.extractMaskedArea);
    this.crop = new CropLayerController(deps.crop);
    this.copy = new CopyLayerController(deps.copy);
    this.newRasterLayer = new NewRasterLayerController(deps.newRasterLayer);
    this.layers = {
      applyStructuralPreview: (action) => (this.disposed ? false : this.structural.preview(action)),
      canCommitStructural: () => this.structural.canCommit(),
      commitGeneratedImageResult: (options) =>
        this.disposed ? Promise.resolve({ status: 'aborted' }) : deps.commitGeneratedImageResult(options),
      commitStructural: (label, forward, inverse) => this.structural.commit(label, forward, inverse),
      invertMask: (layerId) => (this.disposed ? false : this.mask.invert(layerId)),
    };
    this.previews = {
      drawLayerThumbnail: (layerId, target, maxSize) =>
        this.disposed ? false : this.thumbnail.draw(layerId, target, maxSize),
      requestLayerThumbnail: (layerId) => (this.disposed ? Promise.resolve('stale') : this.thumbnail.request(layerId)),
    };
  }

  dispose(): void {
    this.disposed = true;
    this.mask.dispose();
    this.thumbnail.dispose();
    this.structural.dispose();
    this.rasterize.dispose();
    this.merge.dispose();
    this.booleanMerge.dispose();
    this.extractMaskedArea.dispose();
    this.crop.dispose();
    this.copy.dispose();
    this.newRasterLayer.dispose();
  }
}
