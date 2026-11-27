import { GridItem } from '@chakra-ui/react';
import { useDroppable } from '@dnd-kit/core';
import { DropZone } from '@platform/ui/DropZone';
import { DropZoneLabel } from '@workbench/widgets/shared/DropZoneLabel';
import { useTranslation } from 'react-i18next';

import { CANVAS_IMAGE_DROP_DESTINATION_ATTRIBUTE, type CanvasImageDropData } from './canvasImageDnd';

export interface CanvasImageDropZoneProps {
  colSpan: number;
  data: CanvasImageDropData;
  id: string;
  /** A file dragged in from the desktop hovers this zone; dnd-kit reports its own drags. */
  isFileOver?: boolean;
  labelKey:
    | 'widgets.canvas.import.dropControl'
    | 'widgets.canvas.import.dropInpaintMask'
    | 'widgets.canvas.import.dropRaster'
    | 'widgets.canvas.import.dropRegionalReference'
    | 'widgets.canvas.import.dropResizedControl';
  row: number;
}

export const CanvasImageDropZone = ({
  colSpan,
  data,
  id,
  isFileOver = false,
  labelKey,
  row,
}: CanvasImageDropZoneProps) => {
  const { t } = useTranslation();
  const { isOver: isDragOver, setNodeRef } = useDroppable({ data, id });
  const isOver = isDragOver || isFileOver;

  return (
    <GridItem
      ref={setNodeRef}
      colSpan={colSpan}
      gridRow={row}
      data-over={isOver || undefined}
      position="relative"
      {...{ [CANVAS_IMAGE_DROP_DESTINATION_ATTRIBUTE]: data.destination }}
    >
      <DropZone
        alignItems="center"
        display="flex"
        inset="0.5"
        isOver={isOver}
        justifyContent="center"
        opacity={isOver ? 0.98 : 0.88}
        position="absolute"
        variant="overlay"
      >
        <DropZoneLabel>{t(labelKey)}</DropZoneLabel>
      </DropZone>
    </GridItem>
  );
};
