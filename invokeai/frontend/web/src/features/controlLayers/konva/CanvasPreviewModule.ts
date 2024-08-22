import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasProgressImageModule } from 'features/controlLayers/konva/CanvasProgressImageModule';
import Konva from 'konva';

import { CanvasBboxModule } from './CanvasBboxModule';
import { CanvasStagingAreaModule } from './CanvasStagingAreaModule';
import { CanvasToolModule } from './CanvasToolModule';

export class CanvasPreviewModule {
  manager: CanvasManager;

  konva: {
    layer: Konva.Layer;
  };

  tool: CanvasToolModule;
  bbox: CanvasBboxModule;
  stagingArea: CanvasStagingAreaModule;
  progressImage: CanvasProgressImageModule;

  constructor(manager: CanvasManager) {
    this.manager = manager;
    this.konva = {
      layer: new Konva.Layer({ listening: false, imageSmoothingEnabled: false }),
    };

    this.stagingArea = new CanvasStagingAreaModule(this);
    this.konva.layer.add(...this.stagingArea.getNodes());

    this.progressImage = new CanvasProgressImageModule(this);
    this.konva.layer.add(...this.progressImage.getNodes());

    this.bbox = new CanvasBboxModule(this);
    this.konva.layer.add(this.bbox.konva.group);

    this.tool = new CanvasToolModule(this);
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
