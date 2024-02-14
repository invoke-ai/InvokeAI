import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Flex, Icon, IconButton, Spacer, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiInfoBold, PiTrashSimpleBold } from 'react-icons/pi';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { isMouseOverNode, handleMouseOut, handleMouseOver } = useMouseOverNode(nodeId);
  const { t } = useTranslation();
  
  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: `${nodeId}.${fieldName}` });

  const style = {
    transform: CSS.Translate.toString(transform),
    transition,
  };

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
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
    >
      <Flex>
        <EditableFieldTitle nodeId={nodeId} fieldName={fieldName} kind="input" />
        <Spacer />
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
