import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { OutputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldWrapper';
import { useOutputFieldName } from 'features/nodes/hooks/useOutputFieldName';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const OutputFieldUnknownPlaceholder = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const name = useOutputFieldName(nodeId, fieldName);

  return (
    <OutputFieldWrapper>
      <FormControl isInvalid={true} alignItems="stretch" justifyContent="space-between" gap={2} h="full" w="full">
        <FormLabel display="flex" alignItems="center" h="full" color="error.300" mb={0} px={1} gap={2}>
          {t('nodes.unknownOutput', { name })}
        </FormLabel>
      </FormControl>
    </OutputFieldWrapper>
  );
});

OutputFieldUnknownPlaceholder.displayName = 'OutputFieldUnknownPlaceholder';
