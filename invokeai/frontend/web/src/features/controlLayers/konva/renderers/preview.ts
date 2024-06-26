import Konva from 'konva';

import type { CanvasBbox } from './bbox';
import type { CanvasDocumentSizeOverlay } from './documentSizeOverlay';
import type { CanvasStagingArea } from './stagingArea';
import type { CanvasTool } from './tool';

export class CanvasPreview {
  konvaLayer: Konva.Layer;
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
    this.konvaLayer = new Konva.Layer({ listening: true });

    this.bbox = bbox;
    this.konvaLayer.add(this.bbox.group);

    this.tool = tool;
    this.konvaLayer.add(this.tool.group);

    this.documentSizeOverlay = documentSizeOverlay;
    this.konvaLayer.add(this.documentSizeOverlay.group);

    this.stagingArea = stagingArea;
    this.konvaLayer.add(this.stagingArea.group);
  }
}
