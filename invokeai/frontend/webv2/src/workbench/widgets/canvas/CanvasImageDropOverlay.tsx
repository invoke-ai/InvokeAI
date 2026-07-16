import { Grid } from '@chakra-ui/react';
import { useDndContext } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';

import { getCanvasImageDropData, getCanvasImageDropId } from './canvasImageDnd';
import { CanvasImageDropZone, type CanvasImageDropZoneProps } from './CanvasImageDropZone';

export const CANVAS_IMAGE_DROP_LAYOUT = [
  { destination: 'raster', colSpan: 3, row: 1 },
  { destination: 'control', colSpan: 3, row: 1 },
  { destination: 'regional-reference', colSpan: 2, row: 2 },
  { destination: 'inpaint-mask', colSpan: 2, row: 2 },
  { destination: 'control-resized', colSpan: 2, row: 2 },
] as const;

export const CANVAS_IMAGE_DROP_GRID_LAYOUT = {
  gap: '0',
  padding: '0',
  templateColumns: 'repeat(6, minmax(0, 1fr))',
  templateRows: 'repeat(2, minmax(0, 1fr))',
} as const;

const LABEL_KEYS = {
  control: 'widgets.canvas.import.dropControl',
  'control-resized': 'widgets.canvas.import.dropResizedControl',
  'inpaint-mask': 'widgets.canvas.import.dropInpaintMask',
  raster: 'widgets.canvas.import.dropRaster',
  'regional-reference': 'widgets.canvas.import.dropRegionalReference',
} as const satisfies Record<
  (typeof CANVAS_IMAGE_DROP_LAYOUT)[number]['destination'],
  CanvasImageDropZoneProps['labelKey']
>;

const CANVAS_IMAGE_DROP_ZONES = CANVAS_IMAGE_DROP_LAYOUT.map((zone) => ({
  ...zone,
  data: getCanvasImageDropData(zone.destination),
  id: getCanvasImageDropId(zone.destination),
  labelKey: LABEL_KEYS[zone.destination],
})) satisfies readonly CanvasImageDropZoneProps[];

export const CanvasImageDropOverlay = ({
  isDocumentEditingLocked,
  isInteractionLocked,
}: {
  isDocumentEditingLocked: boolean;
  isInteractionLocked: boolean;
}) => {
  const { active } = useDndContext();

  if (isDocumentEditingLocked || isInteractionLocked || !isGalleryImageDragData(active?.data.current)) {
    return null;
  }

  return (
    <Grid {...CANVAS_IMAGE_DROP_GRID_LAYOUT} inset="0" pointerEvents="none" position="absolute" zIndex="2">
      {CANVAS_IMAGE_DROP_ZONES.map((zone) => (
        <CanvasImageDropZone key={zone.id} {...zone} />
      ))}
    </Grid>
  );
};
