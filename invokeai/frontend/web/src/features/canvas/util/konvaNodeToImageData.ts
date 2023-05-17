import Konva from 'konva';
import { IRect } from 'konva/lib/types';
import { dataURLToImageData } from './dataURLToImageData';

/**
 * Converts a Konva node to an ImageData object
 * @param node - The Konva node to convert to an ImageData object
 * @param boundingBox - The bounding box to crop to
 * @returns A Promise that resolves with ImageData object of the node cropped to the bounding box
 */
export const konvaNodeToImageData = async (
  node: Konva.Node,
  boundingBox: IRect
): Promise<ImageData> => {
  // get a dataURL of the bbox'd region
  const dataURL = node.toDataURL(boundingBox);

  return await dataURLToImageData(
    dataURL,
    boundingBox.width,
    boundingBox.height
  );
};
