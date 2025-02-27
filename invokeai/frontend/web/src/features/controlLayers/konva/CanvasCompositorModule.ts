import { withResult, withResultAsync } from 'common/util/result';
import { CanvasCacheModule } from 'features/controlLayers/konva/CanvasCacheModule';
import type { CanvasEntityAdapter, CanvasEntityAdapterFromType } from 'features/controlLayers/konva/CanvasEntity/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleBase } from 'features/controlLayers/konva/CanvasModuleBase';
import type { Transparency } from 'features/controlLayers/konva/util';
import {
  canvasToBlob,
  canvasToImageData,
  getImageDataTransparency,
  getPrefixedId,
  getRectUnion,
  mapId,
  previewBlob,
} from 'features/controlLayers/konva/util';
import {
  selectActiveControlLayerEntities,
  selectActiveInpaintMaskEntities,
  selectActiveRasterLayerEntities,
  selectActiveRegionalGuidanceEntities,
} from 'features/controlLayers/store/selectors';
import type {
  CanvasRenderableEntityIdentifier,
  CanvasRenderableEntityState,
  CanvasRenderableEntityType,
  GenerationMode,
  Rect,
} from 'features/controlLayers/store/types';
import { getEntityIdentifier } from 'features/controlLayers/store/types';
import { imageDTOToImageObject } from 'features/controlLayers/store/util';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { atom, computed } from 'nanostores';
import type { Logger } from 'roarr';
import { serializeError } from 'serialize-error';
import { getImageDTOSafe, uploadImage } from 'services/api/endpoints/images';
import type { ImageDTO, UploadImageArg } from 'services/api/types';
import stableHash from 'stable-hash';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';
import type { JsonObject, SetOptional } from 'type-fest';

type CompositingOptions = {
  /**
   * The global composite operation to use when compositing each entity.
   * See: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation
   */
  globalCompositeOperation?: GlobalCompositeOperation;
};

/**
 * Handles compositing operations:
 * - Rasterizing and uploading the composite raster layer
 * - Rasterizing and uploading the composite inpaint mask
 * - Caclulating the generation mode (which requires the composite raster layer and inpaint mask)
 */
export class CanvasCompositorModule extends CanvasModuleBase {
  readonly type = 'compositor';
  readonly id: string;
  readonly path: string[];
  readonly log: Logger;
  readonly parent: CanvasManager;
  readonly manager: CanvasManager;

  $isCompositing = atom(false);
  $isProcessing = atom(false);
  $isUploading = atom(false);
  $isBusy = computed(
    [this.$isCompositing, this.$isProcessing, this.$isUploading],
    (isCompositing, isProcessing, isUploading) => {
      return isCompositing || isProcessing || isUploading;
    }
  );

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId('canvas_compositor');
    this.parent = manager;
    this.manager = manager;
    this.path = this.manager.buildPath(this);
    this.log = this.manager.buildLogger(this);
    this.log.debug('Creating compositor module');
  }

  /**
   * Gets the rect union of all visible entities of the given entity type. This is used for "merge visible".
   *
   * If no entity type is provided, all visible entities are included in the rect.
   *
   * @param type The optional entity type
   * @returns The rect
   */
  getVisibleRectOfType = (type?: CanvasRenderableEntityType): Rect => {
    const rects = [];

    for (const adapter of this.manager.getAllAdapters()) {
      if (!adapter.state.isEnabled) {
        continue;
      }
      if (type && adapter.state.type !== type) {
        continue;
      }
      if (adapter.renderer.hasObjects()) {
        rects.push(adapter.transformer.getRelativeRect());
      }
    }

    return getRectUnion(...rects);
  };

  /**
   * Gets the rect union of the given entity adapters. This is used for "merge down" and "merge selected".
   *
   * Unlike `getVisibleRectOfType`, **disabled entities are included in the rect**, per the conventional behaviour of
   * these merge methods.
   *
   * @param adapters The entity adapters to include in the rect
   * @returns The rect
   */
  getRectOfAdapters = (adapters: CanvasEntityAdapter[]): Rect => {
    const rects = [];

    for (const adapter of adapters) {
      if (adapter.renderer.hasObjects()) {
        rects.push(adapter.transformer.getRelativeRect());
      }
    }

    return getRectUnion(...rects);
  };

  /**
   * Gets all visible adapters for the given entity type. Visible adapters are those that are not disabled and have
   * objects to render. This is used for "merge visible" functionality and for calculating the generation mode.
   *
   * This includes all adapters that are not disabled and have objects to render.
   *
   * @param type The entity type
   * @returns The adapters for the given entity type that are eligible to be included in a composite
   */
  getVisibleAdaptersOfType = <T extends CanvasRenderableEntityType>(type: T): CanvasEntityAdapterFromType<T>[] => {
    let entities: CanvasRenderableEntityState[];

    switch (type) {
      case 'raster_layer':
        entities = this.manager.stateApi.getRasterLayersState().entities;
        break;
      case 'inpaint_mask':
        entities = this.manager.stateApi.getInpaintMasksState().entities;
        break;
      case 'control_layer':
        entities = this.manager.stateApi.getControlLayersState().entities;
        break;
      case 'regional_guidance':
        entities = this.manager.stateApi.getRegionsState().entities;
        break;
      default:
        assert(false, `Unhandled entity type: ${type}`);
    }

    const adapters: CanvasEntityAdapter[] = entities
      // Get the identifier for each entity
      .map((entity) => getEntityIdentifier(entity))
      // Get the adapter for each entity
      .map(this.manager.getAdapter)
      // Filter out null adapters
      .filter((adapter) => !!adapter)
      // Filter out adapters that are disabled or have no objects (and are thus not to be included in the composite)
      .filter((adapter) => !adapter.$isDisabled.get() && adapter.renderer.hasObjects());

    return adapters as CanvasEntityAdapterFromType<T>[];
  };

  getCompositeHash = (adapters: CanvasEntityAdapter[], extra: JsonObject): string => {
    const adapterHashes: JsonObject[] = [];

    for (const adapter of adapters) {
      adapterHashes.push(adapter.getHashableState());
    }

    const data: JsonObject = {
      extra,
      adapterHashes,
    };

    return stableHash(data);
  };

  /**
   * Composites the given canvas entities for the given rect and returns the resulting canvas.
   *
   * The canvas element is cached to avoid recomputing it when the canvas state has not changed.
   *
   * The canvas entities are drawn in the order they are provided.
   *
   * @param adapters The adapters for the canvas entities to composite, in the order they should be drawn
   * @param rect The region to include in the canvas
   * @param compositingOptions Options for compositing the entities
   * @returns The composite canvas
   */
  getCompositeCanvas = (
    adapters: CanvasEntityAdapter[],
    rect: Rect,
    compositingOptions?: CompositingOptions
  ): HTMLCanvasElement => {
    const entityIdentifiers = adapters.map((adapter) => adapter.entityIdentifier);

    const hash = this.getCompositeHash(adapters, { rect });
    const cachedCanvas = this.manager.cache.canvasElementCache.get(hash);

    if (cachedCanvas) {
      this.log.debug({ entityIdentifiers, rect }, 'Using cached composite canvas');
      return cachedCanvas;
    }

    this.log.debug({ entityIdentifiers, rect }, 'Building composite canvas');
    this.$isCompositing.set(true);

    const canvas = document.createElement('canvas');
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext('2d');
    assert(ctx !== null, 'Canvas 2D context is null');

    ctx.imageSmoothingEnabled = false;

    if (compositingOptions?.globalCompositeOperation) {
      ctx.globalCompositeOperation = compositingOptions.globalCompositeOperation;
    }

    for (const adapter of adapters) {
      this.log.debug({ entityIdentifier: adapter.entityIdentifier }, 'Drawing entity to composite canvas');
      const adapterCanvas = adapter.getCanvas(rect);
      ctx.drawImage(adapterCanvas, 0, 0);
    }
    this.manager.cache.canvasElementCache.set(hash, canvas);
    this.$isCompositing.set(false);
    return canvas;
  };

  /**
   * Composites the given canvas entities for the given rect and uploads the resulting image.
   *
   * The uploaded image is cached to avoid recomputing it when the canvas state has not changed. The canvas elements
   * created for each entity are also cached to avoid recomputing them when the canvas state has not changed.
   *
   * The canvas entities are drawn in the order they are provided.
   *
   * @param adapters The adapters for the canvas entities to composite, in the order they should be drawn
   * @param rect The region to include in the rasterized image
   * @param uploadOptions Options for uploading the image
   * @param compositingOptions Options for compositing the entities
   * @param forceUpload If true, the image is always re-uploaded, returning a new image DTO
   * @returns A promise that resolves to the image DTO
   */
  getCompositeImageDTO = async (
    adapters: CanvasEntityAdapter[],
    rect: Rect,
    uploadOptions: SetOptional<Omit<UploadImageArg, 'file'>, 'image_category'>,
    compositingOptions?: CompositingOptions,
    forceUpload?: boolean
  ): Promise<ImageDTO> => {
    assert(rect.width > 0 && rect.height > 0, 'Unable to rasterize empty rect');

    const hash = this.getCompositeHash(adapters, { rect });
    const cachedImageName = forceUpload ? undefined : this.manager.cache.imageNameCache.get(hash);

    let imageDTO: ImageDTO | null = null;

    if (cachedImageName) {
      imageDTO = await getImageDTOSafe(cachedImageName);
      if (imageDTO) {
        this.log.debug({ rect, imageName: cachedImageName, imageDTO }, 'Using cached composite image');
        return imageDTO;
      }
      this.log.warn({ rect, imageName: cachedImageName }, 'Cached image name not found, recompositing');
    }

    const getCompositeCanvasResult = withResult(() => this.getCompositeCanvas(adapters, rect, compositingOptions));

    if (getCompositeCanvasResult.isErr()) {
      this.log.error({ error: serializeError(getCompositeCanvasResult.error) }, 'Failed to get composite canvas');
      throw getCompositeCanvasResult.error;
    }

    this.$isProcessing.set(true);
    const blobResult = await withResultAsync(() => canvasToBlob(getCompositeCanvasResult.value));
    this.$isProcessing.set(false);

    if (blobResult.isErr()) {
      this.log.error({ error: serializeError(blobResult.error) }, 'Failed to convert composite canvas to blob');
      throw blobResult.error;
    }
    const blob = blobResult.value;

    if (this.manager._isDebugging) {
      previewBlob(blob, 'Composite');
    }

    this.$isUploading.set(true);
    const uploadResult = await withResultAsync(() =>
      uploadImage({
        file: new File([blob], 'canvas-composite.png', { type: 'image/png' }),
        image_category: 'general',
        ...uploadOptions,
      })
    );
    this.$isUploading.set(false);
    if (uploadResult.isErr()) {
      throw uploadResult.error;
    }
    imageDTO = uploadResult.value;
    this.manager.cache.imageNameCache.set(hash, imageDTO.image_name);
    return imageDTO;
  };

  /**
   * Creates a merged composite image from the given entities. The entities are drawn in the order they are provided.
   *
   * The merged image is uploaded to the server and a new entity is created with the uploaded image as the only object.
   *
   * All entities must have the same type.
   *
   * @param entityIdentifiers The entity identifiers to merge
   * @param deleteMergedEntities Whether to delete the merged entities after creating the new merged entity
   * @returns A promise that resolves to the image DTO, or null if the merge failed
   */
  mergeByEntityIdentifiers = async <T extends CanvasRenderableEntityIdentifier>(
    entityIdentifiers: T[],
    deleteMergedEntities: boolean
  ): Promise<ImageDTO | null> => {
    toast({ id: 'MERGE_LAYERS_TOAST', title: t('controlLayers.mergingLayers'), withCount: false });
    if (entityIdentifiers.length <= 1) {
      this.log.warn({ entityIdentifiers }, 'Cannot merge less than 2 entities');
      return null;
    }
    const type = entityIdentifiers[0]?.type;
    assert(type, 'Cannot merge entities with no type (this should never happen)');

    const adapters = this.manager.getAdapters(entityIdentifiers);
    assert(adapters.length === entityIdentifiers.length, 'Failed to get all adapters for entity identifiers');

    const rect = this.getRectOfAdapters(adapters);

    const compositingOptions: CompositingOptions = {
      globalCompositeOperation: type === 'control_layer' ? 'lighter' : undefined,
    };

    const result = await withResultAsync(() =>
      this.getCompositeImageDTO(adapters, rect, { is_intermediate: true }, compositingOptions)
    );

    if (result.isErr()) {
      this.log.error({ error: serializeError(result.error) }, 'Failed to merge selected entities');
      toast({
        id: 'MERGE_LAYERS_TOAST',
        title: t('controlLayers.mergeVisibleError'),
        status: 'error',
        withCount: false,
      });
      return null;
    }

    // All layer types have the same arg - create a new entity with the image as the only object, positioned at the
    // top left corner of the visible rect for the given entity type.
    const addEntityArg = {
      isSelected: true,
      overrides: {
        objects: [imageDTOToImageObject(result.value)],
        position: { x: Math.floor(rect.x), y: Math.floor(rect.y) },
      },
      mergedEntitiesToDelete: deleteMergedEntities ? entityIdentifiers.map(mapId) : [],
    };

    switch (type) {
      case 'raster_layer':
        this.manager.stateApi.addRasterLayer(addEntityArg);
        break;
      case 'inpaint_mask':
        this.manager.stateApi.addInpaintMask(addEntityArg);
        break;
      case 'regional_guidance':
        this.manager.stateApi.addRegionalGuidance(addEntityArg);
        break;
      case 'control_layer':
        this.manager.stateApi.addControlLayer(addEntityArg);
        break;
      default:
        assert<Equals<typeof type, never>>(false, 'Unsupported type for merge');
    }

    toast({ id: 'MERGE_LAYERS_TOAST', title: t('controlLayers.mergeVisibleOk'), status: 'success', withCount: false });

    return result.value;
  };

  /**
   * Merges all visible entities of the given type. This is used for "merge visible" functionality.
   *
   * @param type The type of entity to merge
   * @returns A promise that resolves to the image DTO, or null if the merge failed
   */
  mergeVisibleOfType = (type: CanvasRenderableEntityType): Promise<ImageDTO | null> => {
    let entities: CanvasRenderableEntityState[];

    switch (type) {
      case 'raster_layer':
        entities = this.manager.stateApi.runSelector(selectActiveRasterLayerEntities);
        break;
      case 'inpaint_mask':
        entities = this.manager.stateApi.runSelector(selectActiveInpaintMaskEntities);
        break;
      case 'regional_guidance':
        entities = this.manager.stateApi.runSelector(selectActiveRegionalGuidanceEntities);
        break;
      case 'control_layer':
        entities = this.manager.stateApi.runSelector(selectActiveControlLayerEntities);
        break;
      default:
        assert<Equals<typeof type, never>>(false, 'Unsupported type for merge');
    }

    const entityIdentifiers = entities.map(getEntityIdentifier);

    return this.mergeByEntityIdentifiers(entityIdentifiers, false);
  };

  /**
   * Calculates the transparency of the composite of the give adapters.
   * @param adapters The adapters to composite
   * @param rect The region to include in the composite
   * @param hash The hash to use for caching the result
   * @returns A promise that resolves to the transparency of the composite
   */
  getTransparency = (adapters: CanvasEntityAdapter[], rect: Rect, hash: string): Promise<Transparency> => {
    const entityIdentifiers = adapters.map((adapter) => adapter.entityIdentifier);
    const logCtx = { entityIdentifiers, rect };
    return CanvasCacheModule.getWithFallback({
      cache: this.manager.cache.transparencyCalculationCache,
      key: hash,
      getValue: async () => {
        const compositeInpaintMaskCanvas = this.getCompositeCanvas(adapters, rect);

        const compositeInpaintMaskImageData = await CanvasCacheModule.getWithFallback({
          cache: this.manager.cache.imageDataCache,
          key: hash,
          getValue: () => Promise.resolve(canvasToImageData(compositeInpaintMaskCanvas)),
          onHit: () => this.log.trace(logCtx, 'Using cached image data'),
          onMiss: () => this.log.trace(logCtx, 'Calculating image data'),
        });

        return getImageDataTransparency(compositeInpaintMaskImageData);
      },
      onHit: () => this.log.trace(logCtx, 'Using cached transparency'),
      onMiss: () => this.log.trace(logCtx, 'Calculating transparency'),
    });
  };

  /**
   * Calculates the generation mode for the current canvas state. This is determined by the transparency of the
   * composite raster layer and composite inpaint mask:
   * - Composite raster layer is fully transparent -> txt2img
   * - Composite raster layer is partially transparent -> outpainting
   * - Composite raster layer is opaque & composite inpaint mask is fully transparent -> img2img
   * - Composite raster layer is opaque & composite inpaint mask is partially transparent -> inpainting
   *
   * Definitions:
   * - Fully transparent: all pixels have an alpha value of 0.
   * - Partially transparent: at least one pixel with an alpha value of 0 & at least one pixel with an alpha value
   *   greater than 0.
   * - Opaque: all pixels have an alpha value greater than 0.
   *
   * The generation mode is cached to avoid recalculating it when the canvas state has not changed.
   *
   * @returns The generation mode
   */
  getGenerationMode = async (): Promise<GenerationMode> => {
    const { rect } = this.manager.stateApi.getBbox();

    const rasterLayerAdapters = this.manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const compositeRasterLayerHash = this.getCompositeHash(rasterLayerAdapters, { rect });

    const inpaintMaskAdapters = this.manager.compositor.getVisibleAdaptersOfType('inpaint_mask');
    const compositeInpaintMaskHash = this.getCompositeHash(inpaintMaskAdapters, { rect });

    const hash = stableHash({ rect, compositeInpaintMaskHash, compositeRasterLayerHash });
    const cachedGenerationMode = this.manager.cache.generationModeCache.get(hash);

    if (cachedGenerationMode) {
      this.log.debug({ rect, cachedGenerationMode }, 'Using cached generation mode');
      return cachedGenerationMode;
    }

    this.log.debug({ rect }, 'Calculating generation mode');

    this.$isProcessing.set(true);
    const generationModeResult = await withResultAsync(async () => {
      const compositeRasterLayerTransparency = await this.getTransparency(
        rasterLayerAdapters,
        rect,
        compositeRasterLayerHash
      );

      const compositeInpaintMaskTransparency = await this.getTransparency(
        inpaintMaskAdapters,
        rect,
        compositeInpaintMaskHash
      );

      let generationMode: GenerationMode;
      if (compositeRasterLayerTransparency === 'FULLY_TRANSPARENT') {
        // When the initial image is fully transparent, we are always doing txt2img
        generationMode = 'txt2img';
      } else if (compositeRasterLayerTransparency === 'PARTIALLY_TRANSPARENT') {
        // When the initial image is partially transparent, we are always outpainting
        generationMode = 'outpaint';
      } else if (compositeInpaintMaskTransparency === 'FULLY_TRANSPARENT') {
        // compositeLayerTransparency === 'OPAQUE'
        // When the inpaint mask is fully transparent, we are doing img2img
        generationMode = 'img2img';
      } else {
        // Else at least some of the inpaint mask is opaque, so we are inpainting
        generationMode = 'inpaint';
      }

      this.manager.cache.generationModeCache.set(hash, generationMode);
      return generationMode;
    });

    this.$isProcessing.set(false);
    if (generationModeResult.isErr()) {
      this.log.error({ error: serializeError(generationModeResult.error) }, 'Failed to calculate generation mode');
      throw generationModeResult.error;
    }
    return generationModeResult.value;
  };

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
      $isCompositing: this.$isCompositing.get(),
      $isProcessing: this.$isProcessing.get(),
      $isUploading: this.$isUploading.get(),
      $isBusy: this.$isBusy.get(),
    };
  };
}
