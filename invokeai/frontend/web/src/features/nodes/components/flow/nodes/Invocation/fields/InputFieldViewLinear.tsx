import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Circle, Flex, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import { FieldNotesIconButton } from 'features/nodes/components/flow/nodes/Invocation/fields/FieldNotesIconButton';
import FieldResetToInitialLinearViewValueButton from 'features/nodes/components/flow/nodes/Invocation/fields/FieldResetToInitialLinearViewValueButton';
import { useLinearViewFieldDnd } from 'features/nodes/components/sidePanel/workflow/useLinearViewFieldDnd';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold, PiTrashSimpleBold } from 'react-icons/pi';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

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

export const InputFieldViewLinear = memo(({ nodeId, fieldName }: Props) => {
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
        <Flex flexDir="column" w="full">
          <Flex alignItems="center" gap={2}>
            <EditableFieldTitle nodeId={nodeId} fieldName={fieldName} kind="inputs" />
            <Spacer />
            {isMouseOverNode && <Circle me={2} size={2} borderRadius="full" bg="invokeBlue.500" />}
            <FieldNotesIconButton nodeId={nodeId} fieldName={fieldName} />
            <FieldResetToInitialLinearViewValueButton nodeId={nodeId} fieldName={fieldName} />
            <Tooltip
              label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="inputs" />}
              openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
              placement="top"
            >
              <Flex h="full" alignItems="center">
                <Icon fontSize="sm" color="base.300" as={PiInfoBold} />
              </Flex>
            </Tooltip>
            <IconButton
              aria-label={t('nodes.removeLinearView')}
              tooltip={t('nodes.removeLinearView')}
              variant="ghost"
              size="sm"
              onClick={handleRemoveField}
              icon={<PiTrashSimpleBold />}
            />
          </Flex>
          <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
        </Flex>
      </Flex>
      <DndListDropIndicator dndState={dndListState} />
    </Box>
  );
});

InputFieldViewLinear.displayName = 'InputFieldViewLinear';
