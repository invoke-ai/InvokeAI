import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { OutputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/OutputFieldWrapper';
import { useOutputFieldName } from 'features/nodes/hooks/useOutputFieldName';
import { useOutputFieldTemplateExists } from 'features/nodes/hooks/useOutputFieldTemplateExists';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
}>;

export const OutputFieldGate = memo(({ nodeId, fieldName, children }: Props) => {
  const hasTemplate = useOutputFieldTemplateExists(nodeId, fieldName);

  if (!hasTemplate) {
    return <Fallback nodeId={nodeId} fieldName={fieldName} />;
  }

  return children;
});

OutputFieldGate.displayName = 'OutputFieldGate';

const Fallback = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const name = useOutputFieldName(nodeId, fieldName);

  return (
    <OutputFieldWrapper>
      <FormControl isInvalid={true} alignItems="stretch" justifyContent="space-between" gap={2} h="full" w="full">
        <FormLabel display="flex" alignItems="center" h="full" color="error.300" mb={0} px={1} gap={2}>
          {t('nodes.unexpectedField_withName', { name })}
        </FormLabel>
      </FormControl>
    </OutputFieldWrapper>
  );
});

Fallback.displayName = 'Fallback';
