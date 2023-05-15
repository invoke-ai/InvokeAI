import Konva from 'konva';
import { IRect } from 'konva/lib/types';

/**
 * Converts a Konva node to a dataURL
 * @param node - The Konva node to convert to a dataURL
 * @param boundingBox - The bounding box to crop to
 * @returns A dataURL of the node cropped to the bounding box
 */
export const konvaNodeToDataURL = (
  node: Konva.Node,
  boundingBox: IRect
): string => {
  // get a dataURL of the bbox'd region
  return node.toDataURL(boundingBox);
};
