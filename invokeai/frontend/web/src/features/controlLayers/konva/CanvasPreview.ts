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

  constructor(
    bbox: CanvasBbox,
    tool: CanvasTool,
    documentSizeOverlay: CanvasDocumentSizeOverlay,
    stagingArea: CanvasStagingArea
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
  }
}
