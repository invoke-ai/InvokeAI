import type Konva from 'konva';
import type { IRect } from 'konva/lib/types';

import { canvasToBlob } from './canvasToBlob';

/**
 * Converts a Konva node to a Blob
 * @param node - The Konva node to convert to a Blob
 * @param boundingBox - The bounding box to crop to
 * @returns A Promise that resolves with Blob of the node cropped to the bounding box
 */
export const konvaNodeToBlob = async (node: Konva.Node, boundingBox: IRect): Promise<Blob> => {
  return await canvasToBlob(node.toCanvas(boundingBox));
};
