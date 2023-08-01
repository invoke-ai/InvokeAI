import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { memo } from 'react';
import FieldTooltipContent from './FieldTooltipContent';
import InputFieldRenderer from './InputFieldRenderer';

type Props = {
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
};

const LinearViewField = ({
  nodeData,
  nodeTemplate,
  field,
  fieldTemplate,
}: Props) => {
  return (
    <Flex layerStyle="second" sx={{ borderRadius: 'base', w: 'full', p: 2 }}>
      <FormControl as={Flex} sx={{ flexDir: 'column', gap: 1 }}>
        <Tooltip
          label={
            <FieldTooltipContent
              nodeData={nodeData}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
            />
          }
          openDelay={HANDLE_TOOLTIP_OPEN_DELAY}
          placement="top"
          shouldWrapChildren
          hasArrow
        >
          <FormLabel
            sx={{
              mb: 0,
            }}
          >
            {field.label || fieldTemplate.title} (
            {nodeData.label || nodeTemplate.title})
          </FormLabel>
        </Tooltip>
        <InputFieldRenderer
          nodeData={nodeData}
          nodeTemplate={nodeTemplate}
          field={field}
          fieldTemplate={fieldTemplate}
        />
      </FormControl>
    </Flex>
  );
};

export default memo(LinearViewField);
