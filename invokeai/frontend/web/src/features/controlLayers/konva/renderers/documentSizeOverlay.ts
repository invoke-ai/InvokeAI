import { getArbitraryBaseColor } from '@invoke-ai/ui-library';
import { DOCUMENT_FIT_PADDING_PX } from 'features/controlLayers/konva/constants';
import type { CanvasV2State, StageAttrs } from 'features/controlLayers/store/types';
import Konva from 'konva';

export class CanvasDocumentSizeOverlay {
  group: Konva.Group;
  outerRect: Konva.Rect;
  innerRect: Konva.Rect;
  padding: number;

  constructor(padding?: number) {
    this.padding = padding ?? DOCUMENT_FIT_PADDING_PX;
    this.group = new Konva.Group({ id: 'document_overlay_group', listening: false });
    this.outerRect = new Konva.Rect({
      id: 'document_overlay_outer_rect',
      listening: false,
      fill: getArbitraryBaseColor(10),
      opacity: 0.7,
    });
    this.innerRect = new Konva.Rect({
      id: 'document_overlay_inner_rect',
      listening: false,
      fill: 'white',
      globalCompositeOperation: 'destination-out',
    });
    this.group.add(this.outerRect);
    this.group.add(this.innerRect);
  }

  render(stage: Konva.Stage, document: CanvasV2State['document']) {
    this.group.zIndex(0);

    const x = stage.x();
    const y = stage.y();
    const width = stage.width();
    const height = stage.height();
    const scale = stage.scaleX();

    this.outerRect.setAttrs({
      offsetX: x / scale,
      offsetY: y / scale,
      width: width / scale,
      height: height / scale,
    });

    this.innerRect.setAttrs({
      x: 0,
      y: 0,
      width: document.width,
      height: document.height,
    });
  }

  fitToStage(stage: Konva.Stage, document: CanvasV2State['document'], setStageAttrs: (attrs: StageAttrs) => void) {
    // Fit & center the document on the stage
    const width = stage.width();
    const height = stage.height();
    const docWidthWithBuffer = document.width + this.padding * 2;
    const docHeightWithBuffer = document.height + this.padding * 2;
    const scale = Math.min(Math.min(width / docWidthWithBuffer, height / docHeightWithBuffer), 1);
    const x = (width - docWidthWithBuffer * scale) / 2 + this.padding * scale;
    const y = (height - docHeightWithBuffer * scale) / 2 + this.padding * scale;
    stage.setAttrs({ x, y, width, height, scaleX: scale, scaleY: scale });
    setStageAttrs({ x, y, width, height, scale });
  }
}
