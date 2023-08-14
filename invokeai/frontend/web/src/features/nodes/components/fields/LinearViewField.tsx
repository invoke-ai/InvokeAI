import { Flex, FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { HANDLE_TOOLTIP_OPEN_DELAY } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';
import { memo } from 'react';
import FieldTitle from './FieldTitle';
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
  // const dispatch = useAppDispatch();
  // const handleRemoveField = useCallback(() => {
  //   dispatch(
  //     workflowExposedFieldRemoved({
  //       nodeId: nodeData.id,
  //       fieldName: field.name,
  //     })
  //   );
  // }, [dispatch, field.name, nodeData.id]);

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
              display: 'flex',
              justifyContent: 'space-between',
              mb: 0,
            }}
          >
            <FieldTitle
              nodeData={nodeData}
              nodeTemplate={nodeTemplate}
              field={field}
              fieldTemplate={fieldTemplate}
            />
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
