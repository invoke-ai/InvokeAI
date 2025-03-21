import { FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InputFieldWrapper } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldWrapper';
import { useInputFieldInstanceExists } from 'features/nodes/hooks/useInputFieldInstanceExists';
import { useInputFieldNameSafe } from 'features/nodes/hooks/useInputFieldNameSafe';
import { useInputFieldTemplateExists } from 'features/nodes/hooks/useInputFieldTemplateExists';
import type { PropsWithChildren, ReactNode } from 'react';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = PropsWithChildren<{
  nodeId: string;
  fieldName: string;
  fallback?: ReactNode;
  formatLabel?: (name: string) => string;
}>;

export const InputFieldGate = memo(({ nodeId, fieldName, children, fallback, formatLabel }: Props) => {
  const hasInstance = useInputFieldInstanceExists(nodeId, fieldName);
  const hasTemplate = useInputFieldTemplateExists(nodeId, fieldName);

  if (!hasTemplate || !hasInstance) {
    // fallback may be null, indicating we should render nothing at all - must check for undefined explicitly
    if (fallback !== undefined) {
      return fallback;
    }
    return (
      <Fallback
        nodeId={nodeId}
        fieldName={fieldName}
        formatLabel={formatLabel}
        hasInstance={hasInstance}
        hasTemplate={hasTemplate}
      />
    );
  }

  return children;
});

InputFieldGate.displayName = 'InputFieldGate';

const Fallback = memo(
  ({
    nodeId,
    fieldName,
    formatLabel,
    hasTemplate,
    hasInstance,
  }: {
    nodeId: string;
    fieldName: string;
    formatLabel?: (name: string) => string;
    hasTemplate: boolean;
    hasInstance: boolean;
  }) => {
    const { t } = useTranslation();
    const name = useInputFieldNameSafe(nodeId, fieldName);
    const label = useMemo(() => {
      if (formatLabel) {
        return formatLabel(name);
      }
      if (hasTemplate && !hasInstance) {
        return t('nodes.missingField_withName', { name });
      }
      if (!hasTemplate && hasInstance) {
        return t('nodes.unexpectedField_withName', { name });
      }
      return t('nodes.unknownField_withName', { name });
    }, [formatLabel, hasInstance, hasTemplate, name, t]);

    return (
      <InputFieldWrapper>
        <FormControl isInvalid={true} alignItems="stretch" justifyContent="center" gap={2} h="full" w="full">
          <FormLabel display="flex" mb={0} px={1} py={2} gap={2}>
            {label}
          </FormLabel>
        </FormControl>
      </InputFieldWrapper>
    );
  }
);

Fallback.displayName = 'Fallback';
