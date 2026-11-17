import type { NativeFileDrag } from '@platform/react/useNativeFileDrag';

import { Grid } from '@chakra-ui/react';
import { useDndContext } from '@dnd-kit/core';
import { isGalleryImageDragData } from '@features/gallery/utility';
import { DropZone } from '@platform/ui/DropZone';
import { DropZoneLabel } from '@workbench/widgets/shared/DropZoneLabel';
import { useTranslation } from 'react-i18next';

import { getCanvasImageDropData, getCanvasImageDropId, type CanvasImageDropData } from './canvasImageDnd';
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

/**
 * The drop zones over the canvas while a gallery image is dragged with dnd-kit, or files from the desktop
 * (`fileDrag`); both land the same way. A canvas held by a running operation takes neither and says so instead.
 */
export const CanvasImageDropOverlay = ({
  fileDrag,
  isDocumentEditingLocked,
}: {
  fileDrag: NativeFileDrag<CanvasImageDropData['destination']> | null;
  isDocumentEditingLocked: boolean;
}) => {
  const { t } = useTranslation();
  const { active } = useDndContext();

  if (fileDrag === null && !isGalleryImageDragData(active?.data.current)) {
    return null;
  }

  if (isDocumentEditingLocked) {
    return (
      <DropZone
        alignItems="center"
        display="flex"
        inset="0.5"
        isDisabled
        justifyContent="center"
        p="6"
        pointerEvents="none"
        position="absolute"
        variant="overlay"
        zIndex="2"
      >
        <DropZoneLabel>{t('widgets.layers.actions.busy')}</DropZoneLabel>
      </DropZone>
    );
  }

  return (
    <Grid {...CANVAS_IMAGE_DROP_GRID_LAYOUT} inset="0" pointerEvents="none" position="absolute" zIndex="2">
      {CANVAS_IMAGE_DROP_ZONES.map((zone) => (
        <CanvasImageDropZone key={zone.id} {...zone} isFileOver={fileDrag?.zone === zone.destination} />
      ))}
    </Grid>
  );
};
