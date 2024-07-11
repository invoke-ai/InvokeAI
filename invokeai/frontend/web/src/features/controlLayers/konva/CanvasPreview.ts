import type { CanvasProgressPreview } from 'features/controlLayers/konva/CanvasProgressPreview';
import Konva from 'konva';

import type { CanvasBbox } from './CanvasBbox';
import type { CanvasDocumentSizeOverlay } from './CanvasDocumentSizeOverlay';
import type { CanvasStagingArea } from './CanvasStagingArea';
import type { CanvasTool } from './CanvasTool';

export class CanvasPreview {
  layer: Konva.Layer;
  tool: CanvasTool;
  bbox: CanvasBbox;
  documentSizeOverlay: CanvasDocumentSizeOverlay;
  stagingArea: CanvasStagingArea;
  progressPreview: CanvasProgressPreview;

  constructor(
    bbox: CanvasBbox,
    tool: CanvasTool,
    documentSizeOverlay: CanvasDocumentSizeOverlay,
    stagingArea: CanvasStagingArea,
    progressPreview: CanvasProgressPreview
  ) {
    this.layer = new Konva.Layer({ listening: true, imageSmoothingEnabled: false });

    this.documentSizeOverlay = documentSizeOverlay;
    this.layer.add(this.documentSizeOverlay.group);

    this.stagingArea = stagingArea;
    this.layer.add(this.stagingArea.group);

    this.bbox = bbox;
    this.layer.add(this.bbox.group);

    this.tool = tool;
    this.layer.add(this.tool.group);

    this.progressPreview = progressPreview;
    this.layer.add(this.progressPreview.group);
  }
}
