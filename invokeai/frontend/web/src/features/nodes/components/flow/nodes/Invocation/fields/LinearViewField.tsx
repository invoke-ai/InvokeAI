import { Flex, Icon, Spacer } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import NodeSelectionOverlay from 'common/components/NodeSelectionOverlay';
import { useMouseOverNode } from 'features/nodes/hooks/useMouseOverNode';
import { workflowExposedFieldRemoved } from 'features/nodes/store/workflowSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaInfoCircle, FaTrash } from 'react-icons/fa';

import EditableFieldTitle from './EditableFieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { isMouseOverNode, handleMouseOut, handleMouseOver } =
    useMouseOverNode(nodeId);
  const { t } = useTranslation();
  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

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
        <EditableFieldTitle
          nodeId={nodeId}
          fieldName={fieldName}
          kind="input"
        />
        <Spacer />
        <InvTooltip
          label={
            <FieldTooltipContent
              nodeId={nodeId}
              fieldName={fieldName}
              kind="input"
            />
          }
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
        >
          <Flex h="full" alignItems="center">
            <Icon as={FaInfoCircle} />
          </Flex>
        </InvTooltip>
        <InvIconButton
          aria-label={t('nodes.removeLinearView')}
          tooltip={t('nodes.removeLinearView')}
          variant="ghost"
          size="sm"
          onClick={handleRemoveField}
          icon={<FaTrash />}
        />
      </Flex>
      <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      <NodeSelectionOverlay isSelected={false} isHovered={isMouseOverNode} />
    </Flex>
  );
};

export default memo(LinearViewField);
