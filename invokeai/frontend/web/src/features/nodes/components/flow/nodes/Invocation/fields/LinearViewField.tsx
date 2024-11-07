import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Circle, Flex, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { DndListDropIndicator } from 'features/dnd/DndListDropIndicator';
import { InvocationInputFieldCheck } from 'features/nodes/components/flow/nodes/Invocation/fields/InvocationFieldCheck';
import { useLinearViewFieldDnd } from 'features/nodes/components/sidePanel/workflow/useLinearViewFieldDnd';
import { useFieldOriginalValue } from 'features/nodes/hooks/useFieldOriginalValue';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import type { FieldIdentifier } from 'features/nodes/types/field';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiInfoBold, PiTrashSimpleBold } from 'react-icons/pi';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  fieldIdentifier: FieldIdentifier;
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

const LinearViewFieldInternal = ({ fieldIdentifier }: Props) => {
  const dispatch = useAppDispatch();
  const { isValueChanged, onReset } = useFieldOriginalValue(fieldIdentifier.nodeId, fieldIdentifier.fieldName);
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(fieldIdentifier.nodeId);
  const { t } = useTranslation();

  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved(fieldIdentifier));
  }, [dispatch, fieldIdentifier]);

  const ref = useRef<HTMLDivElement>(null);
  const [dndListState, isDragging] = useLinearViewFieldDnd(ref, fieldIdentifier);

  return (
    <Box position="relative" w="full">
      <Flex
        ref={ref}
        // This is used to trigger the post-move flash animation
        data-field-name={`${fieldIdentifier.nodeId}-${fieldIdentifier.fieldName}`}
        data-is-dragging={isDragging}
        onMouseEnter={handleMouseOver}
        onMouseLeave={handleMouseOut}
        sx={sx}
      >
        <Flex flexDir="column" w="full">
          <Flex alignItems="center" gap={2}>
            <EditableFieldTitle nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} kind="inputs" />
            <Spacer />
            {isMouseOverNode && <Circle me={2} size={2} borderRadius="full" bg="invokeBlue.500" />}
            {isValueChanged && (
              <IconButton
                aria-label={t('nodes.resetToDefaultValue')}
                tooltip={t('nodes.resetToDefaultValue')}
                variant="ghost"
                size="sm"
                onClick={onReset}
                icon={<PiArrowCounterClockwiseBold />}
              />
            )}
            <Tooltip
              label={
                <FieldTooltipContent
                  nodeId={fieldIdentifier.nodeId}
                  fieldName={fieldIdentifier.fieldName}
                  kind="inputs"
                />
              }
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
          <InputFieldRenderer nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
        </Flex>
      </Flex>
      <DndListDropIndicator dndState={dndListState} />
    </Box>
  );
};

const LinearViewField = ({ fieldIdentifier }: Props) => {
  return (
    <InvocationInputFieldCheck nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
      <LinearViewFieldInternal fieldIdentifier={fieldIdentifier} />
    </InvocationInputFieldCheck>
  );
};

export default memo(LinearViewField);
