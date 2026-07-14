import type { NumberInput as ChakraNumberInput } from '@chakra-ui/react';

import { HStack, NumberInput, Text } from '@chakra-ui/react';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

interface SelectedTransform {
  id: string;
  x: number;
  y: number;
}

/**
 * Move tool options: numeric X / Y position of the selected layer (document
 * pixels). Reads the committed transform from the reducer and writes each edit
 * back through the engine's structural history (so it shares the canvas undo
 * stack with drags and nudges). Disabled with no selection.
 */
export const MoveOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const selected = useActiveProjectSelector(
    (project): SelectedTransform | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId
        ? document.layers.find((entry) => entry.id === document.selectedLayerId)
        : undefined;
      return layer ? { id: layer.id, x: layer.transform.x, y: layer.transform.y } : null;
    },
    (a, b) => a?.id === b?.id && a?.x === b?.x && a?.y === b?.y
  );

  const commitAxis = useCallback(
    (axis: 'x' | 'y', next: number) => {
      if (!selected || next === selected[axis]) {
        return;
      }
      const forwardTransform = axis === 'x' ? { x: next } : { y: next };
      const inverseTransform = axis === 'x' ? { x: selected.x } : { y: selected.y };
      engine.layers.commitStructural(
        t('widgets.canvas.toolOptions.movePosition'),
        { id: selected.id, patch: { transform: forwardTransform }, type: 'updateCanvasLayer' },
        { id: selected.id, patch: { transform: inverseTransform }, type: 'updateCanvasLayer' }
      );
    },
    [engine, selected, t]
  );

  const onXChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        commitAxis('x', Math.round(valueAsNumber));
      }
    },
    [commitAxis]
  );

  const onYChange = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        commitAxis('y', Math.round(valueAsNumber));
      }
    },
    [commitAxis]
  );

  const disabled = !selected;

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionX')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={selected ? String(Math.round(selected.x)) : ''}
          w="5rem"
          onValueChange={onXChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionX')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionY')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={selected ? String(Math.round(selected.y)) : ''}
          w="5rem"
          onValueChange={onYChange}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionY')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
    </HStack>
  );
};
