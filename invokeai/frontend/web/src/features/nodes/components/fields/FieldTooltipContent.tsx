import { Flex, Text } from '@chakra-ui/react';
import { FIELDS } from 'features/nodes/types/constants';
import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
  OutputFieldTemplate,
  OutputFieldValue,
  isInputFieldTemplate,
  isInputFieldValue,
} from 'features/nodes/types/types';
import { startCase } from 'lodash-es';

interface Props {
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: InputFieldValue | OutputFieldValue;
  fieldTemplate: InputFieldTemplate | OutputFieldTemplate;
}

const FieldTooltipContent = ({ field, fieldTemplate }: Props) => {
  const isInputTemplate = isInputFieldTemplate(fieldTemplate);

  return (
    <Flex sx={{ flexDir: 'column' }}>
      <Text sx={{ fontWeight: 600 }}>
        {isInputFieldValue(field) && field.label
          ? `${field.label} (${fieldTemplate.title})`
          : fieldTemplate.title}
      </Text>
      <Text sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
        {fieldTemplate.description}
      </Text>
      <Text>Type: {FIELDS[fieldTemplate.type].title}</Text>
      {isInputTemplate && <Text>Input: {startCase(fieldTemplate.input)}</Text>}
    </Flex>
  );
};

export default FieldTooltipContent;
