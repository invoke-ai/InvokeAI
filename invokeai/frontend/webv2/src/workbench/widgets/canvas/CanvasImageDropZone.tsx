import { Flex, GridItem, Text } from '@chakra-ui/react';
import { useDroppable } from '@dnd-kit/core';
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
      <Flex
        alignItems="center"
        bg={isOver ? 'accent.subtle' : 'bg.muted'}
        borderColor={isOver ? 'accent.solid' : 'border.emphasized'}
        borderStyle="dashed"
        borderWidth="2px"
        inset="0.5"
        justifyContent="center"
        opacity={isOver ? 0.98 : 0.88}
        position="absolute"
        rounded="md"
        shadow={isOver ? '0 0 0 1px {colors.accent.solid}' : undefined}
        transition="background var(--wb-motion-duration-fast) ease, border-color var(--wb-motion-duration-fast) ease, opacity var(--wb-motion-duration-fast) ease, box-shadow var(--wb-motion-duration-fast) ease"
      >
        <Text color="fg" fontSize="sm" fontWeight="700" textAlign="center">
          {t(labelKey)}
        </Text>
      </Flex>
    </GridItem>
  );
};
