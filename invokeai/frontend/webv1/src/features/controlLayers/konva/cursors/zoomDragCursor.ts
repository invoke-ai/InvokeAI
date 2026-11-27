import type { Property } from 'csstype';

import zoomDragCursor from './zoom-drag-cursor.svg?raw';

export const ZOOM_DRAG_CURSOR =
  `url("data:image/svg+xml,${encodeURIComponent(zoomDragCursor)}") 13 10, ns-resize` as Property.Cursor;
