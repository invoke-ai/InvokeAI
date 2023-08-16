import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import { memo } from 'react';
import FieldTitle from './FieldTitle';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeId: string;
  fieldName: string;
};

const LinearViewField = ({ nodeId, fieldName }: Props) => {
  return (
    <Flex
      layerStyle="second"
      sx={{
        position: 'relative',
        borderRadius: 'base',
        w: 'full',
        p: 2,
      }}
    >
      <FormControl as={Flex} sx={{ flexDir: 'column', gap: 1, flexShrink: 1 }}>
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
          shouldWrapChildren
          hasArrow
        >
          <FormLabel
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              mb: 0,
            }}
          >
            <FieldTitle nodeId={nodeId} fieldName={fieldName} kind="input" />
          </FormLabel>
        </Tooltip>
        <InputFieldRenderer nodeId={nodeId} fieldName={fieldName} />
      </FormControl>
    </Flex>
  );
};

export default memo(LinearViewField);
