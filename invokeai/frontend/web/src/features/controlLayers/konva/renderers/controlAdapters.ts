import type { EntityToKonvaMap, EntityToKonvaMapping, ImageEntry } from 'features/controlLayers/konva/entityToKonvaMap';
import { LightnessToAlphaFilter } from 'features/controlLayers/konva/filters';
import { CA_LAYER_IMAGE_NAME, CA_LAYER_NAME, CA_LAYER_OBJECT_GROUP_NAME } from 'features/controlLayers/konva/naming';
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
 * Logic for creating and rendering control adapter (control net & t2i adapter) layers. These layers have image objects
 * and require some special handling to update the source and attributes as control images are swapped or processed.
 */

/**
 * Creates a control adapter layer.
 * @param stage The konva stage
 * @param entity The control adapter layer state
 */
const getControlAdapter = (map: EntityToKonvaMap, entity: ControlAdapterEntity): EntityToKonvaMapping => {
  let mapping = map.getMapping(entity.id);
  if (mapping) {
    return mapping;
  }
  const konvaLayer = new Konva.Layer({
    id: entity.id,
    name: CA_LAYER_NAME,
    imageSmoothingEnabled: false,
    listening: false,
  });
  const konvaObjectGroup = createObjectGroup(konvaLayer, CA_LAYER_OBJECT_GROUP_NAME);
  map.stage.add(konvaLayer);
  mapping = map.addMapping(entity.id, konvaLayer, konvaObjectGroup);
  return mapping;
};

/**
 * Renders a control adapter layer. If the layer doesn't already exist, it is created. Otherwise, the layer is updated
 * with the current image source and attributes.
 * @param stage The konva stage
 * @param entity The control adapter layer state
 * @param getImageDTO A function to retrieve an image DTO from the server, used to update the image source
 */
export const renderControlAdapter = async (map: EntityToKonvaMap, entity: ControlAdapterEntity): Promise<void> => {
  const mapping = getControlAdapter(map, entity);
  const imageObject = entity.processedImageObject ?? entity.imageObject;

  if (!imageObject) {
    // The user has deleted/reset the image
    mapping.getEntries().forEach((entry) => {
      mapping.destroyEntry(entry.id);
    });
    return;
  }

  let entry = mapping.getEntries<ImageEntry>()[0];
  const opacity = entity.opacity;
  const visible = entity.isEnabled;
  const filters = entity.filter === 'LightnessToAlphaFilter' ? [LightnessToAlphaFilter] : [];

  if (!entry) {
    entry = await createImageObjectGroup({
      mapping,
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
        entry,
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

export const renderControlAdapters = (map: EntityToKonvaMap, entities: ControlAdapterEntity[]): void => {
  // Destroy nonexistent layers
  for (const mapping of map.getMappings()) {
    if (!entities.find((ca) => ca.id === mapping.id)) {
      map.destroyMapping(mapping.id);
    }
  }
  for (const ca of entities) {
    renderControlAdapter(map, ca);
  }
};
