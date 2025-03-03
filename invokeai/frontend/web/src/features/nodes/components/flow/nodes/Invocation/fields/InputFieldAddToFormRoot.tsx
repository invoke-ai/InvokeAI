import { IconButton } from '@invoke-ai/ui-library';
import { useAddNodeFieldToRoot } from 'features/nodes/components/sidePanel/builder/use-add-node-field-to-root';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldAddToFormRoot = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const addToRoot = useAddNodeFieldToRoot(nodeId, fieldName);

  return (
    <IconButton
      variant="ghost"
      tooltip={t('workflows.builder.addToForm')}
      aria-label={t('workflows.builder.addToForm')}
      icon={<PiPlusBold />}
      pointerEvents="auto"
      size="xs"
      onClick={addToRoot}
    />
  );
});

InputFieldAddToFormRoot.displayName = 'InputFieldAddToFormRoot';
