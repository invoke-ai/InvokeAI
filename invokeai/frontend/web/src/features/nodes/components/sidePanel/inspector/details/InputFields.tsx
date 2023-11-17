import { FormControl, FormLabel, Text } from '@chakra-ui/react';
import { useNodeInputFields } from 'features/nodes/hooks/useNodeInputFields';
import { memo } from 'react';

type Props = { nodeId: string };
const InputFields = ({ nodeId }: Props) => {
  const inputs = useNodeInputFields(nodeId);
  return (
    <div>
      {inputs?.map(({ fieldData, fieldTemplate }) => (
        <FormControl key={fieldData.id}>
          <FormLabel>{fieldData.label || fieldTemplate.title}</FormLabel>
          <Text>{fieldData.type}</Text>
        </FormControl>
      ))}
    </div>
  );
};

export default memo(InputFields);
