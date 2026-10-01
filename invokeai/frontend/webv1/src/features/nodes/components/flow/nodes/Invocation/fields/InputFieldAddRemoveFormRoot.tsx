import { IconButton } from '@invoke-ai/ui-library';
import { useAddRemoveFormElement } from 'features/nodes/components/sidePanel/builder/use-add-remove-form-element';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  fieldName: string;
};

export const InputFieldAddRemoveFormRoot = memo(({ nodeId, fieldName }: Props) => {
  const { t } = useTranslation();
  const { isAddedToRoot, addNodeFieldToRoot, removeNodeFieldFromRoot } = useAddRemoveFormElement(nodeId, fieldName);

  const description = useMemo(() => {
    return isAddedToRoot ? t('workflows.builder.removeFromForm') : t('workflows.builder.addToForm');
  }, [isAddedToRoot, t]);

  const icon = useMemo(() => {
    return isAddedToRoot ? <PiMinusBold /> : <PiPlusBold />;
  }, [isAddedToRoot]);

  const onClick = useCallback(() => {
    return isAddedToRoot ? removeNodeFieldFromRoot() : addNodeFieldToRoot();
  }, [isAddedToRoot, addNodeFieldToRoot, removeNodeFieldFromRoot]);

  return (
    <IconButton
      variant="ghost"
      tooltip={description}
      aria-label={description}
      icon={icon}
      pointerEvents="auto"
      size="xs"
      onClick={onClick}
    />
  );
});

InputFieldAddRemoveFormRoot.displayName = 'InputFieldAddRemoveFormRoot';
