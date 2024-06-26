import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import { createImageObjectGroup, updateImageSource } from 'features/controlLayers/konva/renderers/objects';
import { imageDTOToImageObject, imageDTOToImageWithDims } from 'features/controlLayers/store/types';
import Konva from 'konva';
import { assert } from 'tsafe';

export const createStagingArea = (): KonvaNodeManager['preview']['stagingArea'] => {
  const group = new Konva.Group({ id: 'staging_area_group', listening: false });
  return { group, image: null };
};

export const getRenderStagingArea = async (manager: KonvaNodeManager) => {
  const { getStagingAreaState } = manager.stateApi;
  const stagingArea = getStagingAreaState();

  if (!stagingArea || stagingArea.selectedImageIndex === null) {
    if (manager.preview.stagingArea.image) {
      manager.preview.stagingArea.image.konvaImageGroup.visible(false);
      manager.preview.stagingArea.image = null;
    }
    return;
  }

  if (stagingArea.selectedImageIndex) {
    const imageDTO = stagingArea.images[stagingArea.selectedImageIndex];
    assert(imageDTO, 'Image must exist');
    if (manager.preview.stagingArea.image) {
      if (manager.preview.stagingArea.image.imageName !== imageDTO.image_name) {
        await updateImageSource({
          objectRecord: manager.preview.stagingArea.image,
          image: imageDTOToImageWithDims(imageDTO),
        });
      }
    } else {
      manager.preview.stagingArea.image = await createImageObjectGroup({
        obj: imageDTOToImageObject(imageDTO),
        name: imageDTO.image_name,
      });
    }
  }
};
