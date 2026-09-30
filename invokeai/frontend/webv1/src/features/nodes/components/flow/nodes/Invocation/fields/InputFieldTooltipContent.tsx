import { Flex, ListItem, Text, UnorderedList } from '@invoke-ai/ui-library';
import { startCase } from 'es-toolkit/compat';
import { useInputFieldErrors } from 'features/nodes/hooks/useInputFieldErrors';
import { useInputFieldInstance } from 'features/nodes/hooks/useInputFieldInstance';
import { useInputFieldIsAddedToForm } from 'features/nodes/hooks/useInputFieldIsAddedToForm';
import { useInputFieldTemplateOrThrow } from 'features/nodes/hooks/useInputFieldTemplateOrThrow';
import { useFieldTypeName } from 'features/nodes/hooks/usePrettyFieldType';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

interface Props {
  nodeId: string;
  fieldName: string;
}

export const InputFieldTooltipContent = memo(({ fieldName }: Props) => {
  const { t } = useTranslation();

  const fieldInstance = useInputFieldInstance(fieldName);
  const fieldTemplate = useInputFieldTemplateOrThrow(fieldName);
  const fieldTypeName = useFieldTypeName(fieldTemplate.type);
  const fieldErrors = useInputFieldErrors(fieldName);
  const isAddedToForm = useInputFieldIsAddedToForm(fieldName);

  const fieldTitle = useMemo(() => {
    if (fieldInstance.label && fieldTemplate.title) {
      return `${fieldInstance.label} (${fieldTemplate.title})`;
    }

    if (fieldInstance.label && !fieldTemplate.title) {
      return fieldInstance.label;
    }

    return fieldTemplate.title;
  }, [fieldInstance, fieldTemplate]);

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">
        {fieldTitle}
        {isAddedToForm && ' (added to form)'}
      </Text>
      <Text opacity={0.7} fontStyle="oblique 5deg">
        {fieldTemplate.description}
      </Text>
      <Text>
        {t('parameters.type')}: {fieldTypeName}
      </Text>
      <Text>
        {t('common.input')}: {startCase(fieldTemplate.input)}
      </Text>
      {fieldErrors.length > 0 && (
        <>
          <Text color="error.500">{t('common.error_withCount', { count: fieldErrors.length })}:</Text>
          <UnorderedList>
            {fieldErrors.map(({ issue }) => (
              <ListItem key={issue}>{issue}</ListItem>
            ))}
          </UnorderedList>
        </>
      )}
    </Flex>
  );
});

InputFieldTooltipContent.displayName = 'InputFieldTooltipContent';
