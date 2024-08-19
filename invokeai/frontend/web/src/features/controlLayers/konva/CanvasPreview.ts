import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasProgressImage } from 'features/controlLayers/konva/CanvasProgressImage';
import Konva from 'konva';

import { CanvasBbox } from './CanvasBbox';
import { CanvasStagingArea } from './CanvasStagingArea';
import { CanvasTool } from './CanvasTool';

export class CanvasPreview {
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
  };

  tool: CanvasTool;
  bbox: CanvasBbox;
  stagingArea: CanvasStagingArea;
  progressImage: CanvasProgressImage;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({ listening: false, imageSmoothingEnabled: false }),
    };

    this.stagingArea = new CanvasStagingArea(this);
    this.konva.layer.add(...this.stagingArea.getNodes());

    this.progressImage = new CanvasProgressImage(this);
    this.konva.layer.add(...this.progressImage.getNodes());

    this.bbox = new CanvasBbox(this);
    this.konva.layer.add(this.bbox.konva.group);

    this.tool = new CanvasTool(this);
    this.konva.layer.add(this.tool.konva.group);
  }

  getLayer = () => {
    return this.konva.layer;
  };

  destroy() {
    // this.stagingArea.destroy(); // TODO(psyche): implement destroy
    this.progressImage.destroy();
    // this.bbox.destroy(); // TODO(psyche): implement destroy
    this.tool.destroy();
    this.konva.layer.destroy();
  }
}
