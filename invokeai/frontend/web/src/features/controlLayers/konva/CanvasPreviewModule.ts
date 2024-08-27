import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { CanvasModuleABC } from 'features/controlLayers/konva/CanvasModuleABC';
import { CanvasProgressImageModule } from 'features/controlLayers/konva/CanvasProgressImageModule';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import Konva from 'konva';
import type { Logger } from 'roarr';

import { CanvasBboxModule } from './CanvasBboxModule';
import { CanvasStagingAreaModule } from './CanvasStagingAreaModule';
import { CanvasToolModule } from './CanvasToolModule';

export class CanvasPreviewModule extends CanvasModuleABC {
  readonly type = 'preview';

  id: string;
  path: string[];
  manager: CanvasManager;
  log: Logger;
  subscriptions = new Set<() => void>();

  konva: {
    layer: Konva.Layer;
  };

  tool: CanvasToolModule;
  bbox: CanvasBboxModule;
  stagingArea: CanvasStagingAreaModule;
  progressImage: CanvasProgressImageModule;

  constructor(manager: CanvasManager) {
    super();
    this.id = getPrefixedId(this.type);
    this.manager = manager;
    this.path = this.manager.path.concat(this.id);
    this.log = this.manager.buildLogger(this.getLoggingContext);

    this.log.debug('Creating preview module');

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

  repr = () => {
    return {
      id: this.id,
      type: this.type,
      path: this.path,
    };
  };

  destroy = () => {
    this.log.debug('Destroying preview module');
    this.subscriptions.forEach((unsubscribe) => unsubscribe());
    this.stagingArea.destroy();
    this.progressImage.destroy();
    this.bbox.destroy();
    this.tool.destroy();
    this.konva.layer.destroy();
  };

  getLoggingContext = () => {
    return { ...this.manager.getLoggingContext(), path: this.path.join('.') };
  };
}
