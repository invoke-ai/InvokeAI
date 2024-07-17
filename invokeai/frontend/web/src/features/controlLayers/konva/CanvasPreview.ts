import type { CanvasProgressPreview } from 'features/controlLayers/konva/CanvasProgressPreview';
import Konva from 'konva';

import type { CanvasBbox } from './CanvasBbox';
import type { CanvasStagingArea } from './CanvasStagingArea';
import type { CanvasTool } from './CanvasTool';

export class CanvasPreview {
  layer: Konva.Layer;
  tool: CanvasTool;
  bbox: CanvasBbox;
  stagingArea: CanvasStagingArea;
  progressPreview: CanvasProgressPreview;

  constructor(
    bbox: CanvasBbox,
    tool: CanvasTool,
    stagingArea: CanvasStagingArea,
    progressPreview: CanvasProgressPreview
  ) {
    this.layer = new Konva.Layer({ listening: true, imageSmoothingEnabled: false });

    this.stagingArea = stagingArea;
    this.layer.add(this.stagingArea.konva.group);

    this.bbox = bbox;
    this.layer.add(this.bbox.konva.group);

    this.tool = tool;
    this.layer.add(this.tool.konva.group);

    this.progressPreview = progressPreview;
    this.layer.add(this.progressPreview.konva.group);
  }
}
