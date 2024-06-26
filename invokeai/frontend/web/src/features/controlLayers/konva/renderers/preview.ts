import Konva from 'konva';

import type { CanvasBbox } from './bbox';
import type { CanvasDocumentSizeOverlay } from './documentSizeOverlay';
import type { CanvasStagingArea } from './stagingArea';
import type { CanvasTool } from './tool';

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
    this.layer = new Konva.Layer({ listening: true });

    this.bbox = bbox;
    this.layer.add(this.bbox.group);

    this.tool = tool;
    this.layer.add(this.tool.group);

    this.documentSizeOverlay = documentSizeOverlay;
    this.layer.add(this.documentSizeOverlay.group);

    this.stagingArea = stagingArea;
    this.layer.add(this.stagingArea.group);
  }
}
