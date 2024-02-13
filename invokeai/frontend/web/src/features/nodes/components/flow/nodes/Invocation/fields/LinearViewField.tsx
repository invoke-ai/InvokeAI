import { Flex, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useFieldInstance } from 'features/nodes/hooks/useFieldData';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { fieldValueReset } from 'features/nodes/store/nodesSlice';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiInfoBold, PiTrashSimpleBold } from 'react-icons/pi';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const fieldInstance = useFieldInstance(nodeId, fieldName);
  const { originalExposedFieldValues } = useAppSelector((s) => s.workflow);
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);
  const { t } = useTranslation();

  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const originalValue = useMemo(() => {
    return originalExposedFieldValues.find((originalValues) => originalValues.nodeId === nodeId)?.value;
  }, [originalExposedFieldValues, nodeId]);

  const handleResetField = useCallback(() => {
    dispatch(fieldValueReset({ nodeId, fieldName, value: originalValue }));
  }, [dispatch, fieldName, nodeId, originalValue]);

  return (
    <Flex
      onMouseEnter={handleMouseOver}
      onMouseLeave={handleMouseOut}
      layerStyle="second"
      position="relative"
      borderRadius="base"
      w="full"
      p={4}
      flexDir="column"
    >
      <Flex>
        <EditableFieldTitle nodeId={nodeId} fieldName={fieldName} kind="input" />
        <Spacer />
        {originalValue !== fieldInstance?.value && (
          <IconButton
            aria-label={t('nodes.resetToDefaultValue')}
            tooltip={t('nodes.resetToDefaultValue')}
            variant="ghost"
            size="sm"
            onClick={handleResetField}
            icon={<PiArrowCounterClockwiseBold />}
          />
        )}
        <Tooltip
          label={<FieldTooltipContent nodeId={nodeId} fieldName={fieldName} kind="input" />}
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
      <NodeSelectionOverlay isSelected={false} isHovered={isMouseOverNode} />
    </Flex>
  );
};

export default memo(LinearViewField);
