import { DOCUMENT_FIT_PADDING_PX } from 'features/controlLayers/konva/constants';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';
import type { CanvasV2State, StageAttrs } from 'features/controlLayers/store/types';

/**
 * Gets a function to fit the document to the stage, resetting the stage scale to 100%.
 * If the document is smaller than the stage, the stage scale is increased to fit the document.
 * @param arg.manager The konva node manager
 * @param arg.getDocument A function to get the current document state
 * @param arg.setStageAttrs A function to set the stage attributes
 * @returns A function to fit the document to the stage
 */
export const getFitDocumentToStage =
  (arg: {
    manager: KonvaNodeManager;
    getDocument: () => CanvasV2State['document'];
    setStageAttrs: (stageAttrs: StageAttrs) => void;
  }) =>
  (): void => {
    const { manager, getDocument, setStageAttrs } = arg;
    const document = getDocument();
    // Fit & center the document on the stage
    const width = manager.stage.width();
    const height = manager.stage.height();
    const docWidthWithBuffer = document.width + DOCUMENT_FIT_PADDING_PX * 2;
    const docHeightWithBuffer = document.height + DOCUMENT_FIT_PADDING_PX * 2;
    const scale = Math.min(Math.min(width / docWidthWithBuffer, height / docHeightWithBuffer), 1);
    const x = (width - docWidthWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
    const y = (height - docHeightWithBuffer * scale) / 2 + DOCUMENT_FIT_PADDING_PX * scale;
    manager.stage.setAttrs({ x, y, width, height, scaleX: scale, scaleY: scale });
    setStageAttrs({ x, y, width, height, scale });
  };
