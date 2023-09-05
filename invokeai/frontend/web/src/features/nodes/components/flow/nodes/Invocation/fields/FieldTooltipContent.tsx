import { Flex, Text } from '@chakra-ui/react';
import { useFieldData } from 'features/nodes/hooks/useFieldData';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import { FIELDS } from 'features/nodes/types/constants';
import {
  isInputFieldTemplate,
  isInputFieldValue,
} from 'features/nodes/types/types';
import { startCase } from 'lodash-es';
import { memo, useMemo } from 'react';

interface Props {
  nodeId: string;
  fieldName: string;
  kind: 'input' | 'output';
}

const FieldTooltipContent = ({ nodeId, fieldName, kind }: Props) => {
  const field = useFieldData(nodeId, fieldName);
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, kind);
  const isInputTemplate = isInputFieldTemplate(fieldTemplate);
  const fieldTitle = useMemo(() => {
    if (isInputFieldValue(field)) {
      if (field.label && fieldTemplate?.title) {
        return `${field.label} (${fieldTemplate.title})`;
      }

      if (field.label && !fieldTemplate) {
        return field.label;
      }

      if (!field.label && fieldTemplate) {
        return fieldTemplate.title;
      }

      return 'Unknown Field';
    } else {
      return fieldTemplate?.title || 'Unknown Field';
    }
  }, [field, fieldTemplate]);

  return (
    <Flex sx={{ flexDir: 'column' }}>
      <Text sx={{ fontWeight: 600 }}>{fieldTitle}</Text>
      {fieldTemplate && (
        <Text sx={{ opacity: 0.7, fontStyle: 'oblique 5deg' }}>
          {fieldTemplate.description}
        </Text>
      )}
      {fieldTemplate && <Text>Type: {FIELDS[fieldTemplate.type].title}</Text>}
      {isInputTemplate && <Text>Input: {startCase(fieldTemplate.input)}</Text>}
    </Flex>
  );
};

export default memo(FieldTooltipContent);
