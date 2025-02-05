import { Flex, Text } from '@invoke-ai/ui-library';
import { useOutputFieldTemplate } from 'features/nodes/hooks/useOutputFieldTemplate';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

interface Props {
  nodeId: string;
  fieldName: string;
}

export const OutputFieldTooltipContent = memo(({ nodeId, fieldName }: Props) => {
  const fieldTemplate = useOutputFieldTemplate(nodeId, fieldName);
  const fieldTypeName = useFieldTypeName(fieldTemplate.type);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{fieldTemplate.title}</Text>
      <Text opacity={0.7} fontStyle="oblique 5deg">
        {fieldTemplate.description}
      </Text>
      <Text>
        {t('parameters.type')}: {fieldTypeName}
      </Text>
    </Flex>
  );
});

OutputFieldTooltipContent.displayName = 'OutputFieldTooltipContent';
