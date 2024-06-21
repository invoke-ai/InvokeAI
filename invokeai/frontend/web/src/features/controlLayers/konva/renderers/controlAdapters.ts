import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { CA_LAYER_IMAGE_NAME, CA_LAYER_NAME, CA_LAYER_OBJECT_GROUP_NAME } from 'features/controlLayers/konva/naming';
import type { ImageObjectRecord, KonvaEntityAdapter, KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import {
  createImageObjectGroup,
  createObjectGroup,
  updateImageSource,
} from 'features/controlLayers/konva/renderers/objects';
import type { ControlAdapterEntity } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { isEqual } from 'lodash-es';
import { assert } from 'tsafe';

/**
 * Gets a control adapter entity's konva nodes and entity adapter, creating them if they do not exist.
 * @param manager The konva node manager
 * @param entity The control adapter layer state
 */
const getControlAdapter = (manager: KonvaNodeManager, entity: ControlAdapterEntity): KonvaEntityAdapter => {
  const adapter = manager.get(entity.id);
  if (adapter) {
    return adapter;
  }
  const konvaLayer = new Konva.Layer({
    id: entity.id,
    name: CA_LAYER_NAME,
    imageSmoothingEnabled: false,
    listening: false,
  });
  const konvaObjectGroup = createObjectGroup(konvaLayer, CA_LAYER_OBJECT_GROUP_NAME);
  return manager.add(entity, konvaLayer, konvaObjectGroup);
};

/**
 * Renders a control adapter.
 * @param manager The konva node manager
 * @param entity The control adapter entity state
 */
export const renderControlAdapter = async (manager: KonvaNodeManager, entity: ControlAdapterEntity): Promise<void> => {
  const adapter = getControlAdapter(manager, entity);
  const imageObject = entity.processedImageObject ?? entity.imageObject;

  if (!imageObject) {
    // The user has deleted/reset the image
    adapter.getAll().forEach((entry) => {
      adapter.destroy(entry.id);
    });
    return;
  }

  let entry = adapter.getAll<ImageObjectRecord>()[0];
  const opacity = entity.opacity;
  const visible = entity.isEnabled;
  const filters = entity.filter === 'LightnessToAlphaFilter' ? [LightnessToAlphaFilter] : [];

  if (!entry) {
    entry = await createImageObjectGroup({
      adapter: adapter,
      obj: imageObject,
      name: CA_LAYER_IMAGE_NAME,
      onLoad: (konvaImage) => {
        konvaImage.filters(filters);
        konvaImage.cache();
        konvaImage.opacity(opacity);
        konvaImage.visible(visible);
      },
    });
  } else {
    if (entry.isLoading || entry.isError) {
      return;
    }
    assert(entry.konvaImage, `Image entry ${entry.id} must have a konva image if it is not loading or in error state`);
    const imageSource = entry.konvaImage.image();
    assert(imageSource instanceof HTMLImageElement, `Image source must be an HTMLImageElement`);
    if (imageSource.id !== imageObject.image.name) {
      updateImageSource({
        objectRecord: entry,
        image: imageObject.image,
        onLoad: (konvaImage) => {
          konvaImage.filters(filters);
          konvaImage.cache();
          konvaImage.opacity(opacity);
          konvaImage.visible(visible);
        },
      });
    } else {
      if (!isEqual(entry.konvaImage.filters(), filters)) {
        entry.konvaImage.filters(filters);
        entry.konvaImage.cache();
      }
      entry.konvaImage.opacity(opacity);
      entry.konvaImage.visible(visible);
    }
  }
};

/**
 * Gets a function to render all control adapters.
 * @param manager The konva node manager
 * @returns A function to render all control adapters
 */
export const getRenderControlAdapters = (manager: KonvaNodeManager) => {
  const { getControlAdapterEntityStates } = manager.stateApi;

  function renderControlAdapters(): void {
    const entities = getControlAdapterEntityStates();
    // Destroy nonexistent layers
    for (const adapters of manager.getAll('control_adapter')) {
      if (!entities.find((ca) => ca.id === adapters.id)) {
        manager.destroy(adapters.id);
      }
    }
    for (const entity of entities) {
      renderControlAdapter(manager, entity);
    }
  }

  return renderControlAdapters;
};
