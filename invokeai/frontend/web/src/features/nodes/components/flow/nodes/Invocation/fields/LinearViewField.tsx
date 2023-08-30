import { Flex, FormControl, FormLabel, Icon, Tooltip } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import SelectionOverlay from 'common/components/SelectionOverlay';
import { useIsMouseOverField } from 'features/nodes/hooks/useIsMouseOverField';
import { workflowExposedFieldRemoved } from 'features/nodes/store/nodesSlice';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo, useCallback } from 'react';
import { FaInfoCircle, FaTrash } from 'react-icons/fa';
import FieldTitle from './FieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  const dispatch = useAppDispatch();
  const { isMouseOverField, handleMouseOut, handleMouseOver } =
    useIsMouseOverField(nodeId, fieldName);

  const handleRemoveField = useCallback(() => {
    dispatch(workflowExposedFieldRemoved({ nodeId, fieldName }));
  }, [dispatch, fieldName, nodeId]);

  return (
    <Flex
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      layerStyle="second"
      sx={{
        position: 'relative',
        borderRadius: 'base',
        w: 'full',
        p: 2,
      }}
    >
      <FormControl as={Flex} sx={{ flexDir: 'column', gap: 1, flexShrink: 1 }}>
        <FormLabel
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 0,
          }}
        >
          <FieldTitle nodeId={nodeId} fieldName={fieldName} kind="input" />
          <Tooltip
            label={
              <FieldTooltipContent
                nodeId={nodeId}
                fieldName={fieldName}
                kind="input"
              />
            }
            openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
            placement="top"
            hasArrow
          >
            <Flex h="full" alignItems="center">
              <Icon as={FaInfoCircle} />
            </Flex>
          </Tooltip>
          <IAIIconButton
            aria-label="Remove from Linear View"
            tooltip="Remove from Linear View"
            variant="ghost"
            size="sm"
            onClick={handleRemoveField}
            icon={<FaTrash />}
          />
        </FormLabel>
        <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      </FormControl>
      {/* <SelectionOverlay isSelected={false} isHovered={isMouseOverField} /> */}
    </Flex>
  );
};

export default memo(LinearViewField);
