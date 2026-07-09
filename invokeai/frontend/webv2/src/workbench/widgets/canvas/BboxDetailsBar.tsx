import type { NumberInput as ChakraNumberInput } from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';

import { HStack, NumberInput, Text } from '@chakra-ui/react';
import { useCanvasActiveTool } from '@workbench/widgets/canvas/engineStoreHooks';
import { CanvasOptionsBar } from '@workbench/widgets/canvas/tool-options/CanvasOptionsBar';
import { useBboxEditor } from '@workbench/widgets/canvas/tool-options/useBboxEditor';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const BboxDetailsBar = ({ engine }: { engine: CanvasEngine }) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const { bbox, setHeight, setWidth, setX, setY } = useBboxEditor(engine);

  const onWidthChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setWidth(valueAsNumber);
      }
    },
    [setWidth]
  );
  const onHeightChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setHeight(valueAsNumber);
      }
    },
    [setHeight]
  );
  const onXChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setX(valueAsNumber);
      }
    },
    [setX]
  );
  const onYChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        setY(valueAsNumber);
      }
    },
    [setY]
  );

  if (activeTool !== 'bbox') {
    return null;
  }

  return (
    <CanvasOptionsBar>
      <HStack align="center" gap="3">
        <HStack align="center" gap="1.5">
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.canvas.toolOptions.frameWidth')}
          </Text>
          <NumberInput.Root min={1} size="xs" value={String(bbox.width)} w="5rem" onValueChange={onWidthChange}>
            <NumberInput.Control />
            <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.frameWidth')} fontSize="xs" />
          </NumberInput.Root>
        </HStack>
        <HStack align="center" gap="1.5">
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.canvas.toolOptions.frameHeight')}
          </Text>
          <NumberInput.Root min={1} size="xs" value={String(bbox.height)} w="5rem" onValueChange={onHeightChange}>
            <NumberInput.Control />
            <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.frameHeight')} fontSize="xs" />
          </NumberInput.Root>
        </HStack>
        <HStack align="center" gap="1.5">
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.canvas.toolOptions.positionX')}
          </Text>
          <NumberInput.Root size="xs" value={String(bbox.x)} w="5rem" onValueChange={onXChange}>
            <NumberInput.Control />
            <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionX')} fontSize="xs" />
          </NumberInput.Root>
        </HStack>
        <HStack align="center" gap="1.5">
          <Text color="fg.muted" fontSize="2xs">
            {t('widgets.canvas.toolOptions.positionY')}
          </Text>
          <NumberInput.Root size="xs" value={String(bbox.y)} w="5rem" onValueChange={onYChange}>
            <NumberInput.Control />
            <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionY')} fontSize="xs" />
          </NumberInput.Root>
        </HStack>
      </HStack>
    </CanvasOptionsBar>
  );
};
