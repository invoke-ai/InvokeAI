import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Circle, Flex, IconButton, Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import { InputFieldLinearViewConfigIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldLinearViewConfigIconButton';
import { InputFieldNotesIconButtonEditable } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldNotesIconButtonEditable';
import { InputFieldResetToInitialValueIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldResetToInitialValueIconButton';
import { useLinearViewFieldDnd } from 'features/nodes/components/sidePanel/workflow/useLinearViewFieldDnd';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold } from 'react-icons/pi';

import { InputFieldRenderer } from './InputFieldRenderer';
import { InputFieldTitle } from './InputFieldTitle';

type Props = {
  nodeId: string;
  fieldName: string;
};

const sx = {
  layerStyle: 'second',
  alignItems: 'center',
  position: 'relative',
  borderRadius: 'base',
  w: 'full',
  p: 2,
  '&[data-is-dragging=true]': {
    opacity: 0.3,
  },
  transitionProperty: 'common',
} satisfies SystemStyleObject;

export const InputFieldEditModeLinear = memo(({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);
  const { t } = useTranslation();

  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const ref = useRef<HTMLDivElement>(null);
  const [dndListState, isDragging] = useLinearViewFieldDnd(ref, { nodeId, fieldName });

  return (
    <Box position="relative" w="full">
      <Flex
        ref={ref}
        // This is used to trigger the post-move flash animation
        data-field-name={`${nodeId}-${fieldName}`}
        data-is-dragging={isDragging}
        onMouseEnter={handleMouseOver}
        onMouseLeave={handleMouseOut}
        sx={sx}
      >
        <Flex flexDir="column" w="full" gap={1}>
          <Flex alignItems="center" gap={2}>
            <InputFieldTitle nodeId={nodeId} fieldName={fieldName} />
            <Spacer />
            {isMouseOverNode && <Circle me={2} size={2} borderRadius="full" bg="invokeBlue.500" />}
            <InputFieldLinearViewConfigIconButton nodeId={nodeId} fieldName={fieldName} />
            <InputFieldNotesIconButtonEditable nodeId={nodeId} fieldName={fieldName} />
            <InputFieldResetToInitialValueIconButton nodeId={nodeId} fieldName={fieldName} />
            <IconButton
              aria-label={t('nodes.removeLinearView')}
              tooltip={t('nodes.removeLinearView')}
              variant="ghost"
              size="xs"
              onClick={handleRemoveField}
              icon={<PiMinusBold />}
            />
          </Flex>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </Flex>
      <DndListDropIndicator dndState={dndListState} />
    </Box>
  );
});

InputFieldEditModeLinear.displayName = 'InputFieldEditModeLinear';
