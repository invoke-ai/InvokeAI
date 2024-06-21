import { DOCUMENT_FIT_PADDING_PX } from 'features/controlLayers/konva/constants';
import type { KonvaNodeManager } from 'features/controlLayers/konva/nodeManager';

/**
 * Gets a function to fit the document to the stage, resetting the stage scale to 100%.
 * If the document is smaller than the stage, the stage scale is increased to fit the document.
 * @param manager The konva node manager
 * @returns A function to fit the document to the stage
 */
export const getFitDocumentToStage = (manager: KonvaNodeManager) => {
  function fitDocumentToStage(): void {
    const { getDocument, setStageAttrs } = manager.stateApi;
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
  }

  return fitDocumentToStage;
};

/**
 * Gets a function to fit the stage to its container element. Called during resize events.
 * @param manager The konva node manager
 * @returns A function to fit the stage to its container
 */
export const getFitStageToContainer = (manager: KonvaNodeManager) => {
  const { stage, container } = manager;
  const { setStageAttrs } = manager.stateApi;
  function fitStageToContainer(): void {
    stage.width(container.offsetWidth);
    stage.height(container.offsetHeight);
    setStageAttrs({
      x: stage.x(),
      y: stage.y(),
      width: stage.width(),
      height: stage.height(),
      scale: stage.scaleX(),
    });
    manager.konvaApi.renderBackground();
    manager.konvaApi.renderDocumentOverlay();
  }

  return fitStageToContainer;
};
