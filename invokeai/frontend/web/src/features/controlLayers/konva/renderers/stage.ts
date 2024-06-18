import { DOCUMENT_FIT_PADDING_PX } from 'features/controlLayers/konva/constants';
import type { CanvasV2State, StageAttrs } from 'features/controlLayers/store/types';
import type Konva from 'konva';

export const fitDocumentToStage = (stage: Konva.Stage, document: CanvasV2State['document']): StageAttrs => {
  // Fit & center the document on the stage
  const width = stage.width();
  const height = stage.height();
  const docWidthWithBuffer = document.width + DOCUMENT_FIT_PADDING_PX * 2;
  const docHeightWithBuffer = document.height + DOCUMENT_FIT_PADDING_PX * 2;
  const scale = Math.min(Math.min(width / docWidthWithBuffer, height / docHeightWithBuffer), 1);
  const x = (width - docWidthWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
  const y = (height - docHeightWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
  stage.setAttrs({ x, y, width, height, scaleX: scale, scaleY: scale });
  return { x, y, width, height, scale };
};
