import { GridItem, Text } from '@chakra-ui/react';
import { useDroppable } from '@dnd-kit/core';
import { DropZone } from '@platform/ui/DropZone';
import { useTranslation } from 'react-i18next';

import type { CanvasImageDropData } from './canvasImageDnd';

export interface CanvasImageDropZoneProps {
  colSpan: number;
  data: CanvasImageDropData;
  id: string;
  labelKey:
    | 'widgets.canvas.import.dropControl'
    | 'widgets.canvas.import.dropInpaintMask'
    | 'widgets.canvas.import.dropRaster'
    | 'widgets.canvas.import.dropRegionalReference'
    | 'widgets.canvas.import.dropResizedControl';
  row: number;
}

export const CanvasImageDropZone = ({ colSpan, data, id, labelKey, row }: CanvasImageDropZoneProps) => {
  const { t } = useTranslation();
  const { isOver, setNodeRef } = useDroppable({ data, id });

  return (
    <GridItem ref={setNodeRef} colSpan={colSpan} gridRow={row} position="relative">
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
        <Text color="fg" fontSize="sm" fontWeight="700" textAlign="center">
          {t(labelKey)}
        </Text>
      </DropZone>
    </GridItem>
  );
};
