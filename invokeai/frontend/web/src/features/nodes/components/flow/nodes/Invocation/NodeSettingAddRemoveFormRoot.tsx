import { IconButton } from '@invoke-ai/ui-library';
import { useAddRemoveNodeSettingFormElement } from 'features/nodes/components/sidePanel/builder/use-add-remove-form-element';
import { NO_DRAG_CLASS } from 'features/nodes/types/constants';
import type { NodeSettingName } from 'features/nodes/types/workflow';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMinusBold, PiPlusBold } from 'react-icons/pi';

type Props = {
  nodeId: string;
  setting: NodeSettingName;
};

export const NodeSettingAddRemoveFormRoot = memo(({ nodeId, setting }: Props) => {
  const { t } = useTranslation();
  const { isAddedToRoot, addNodeSettingToRoot, removeNodeSettingFromRoot } = useAddRemoveNodeSettingFormElement(
    nodeId,
    setting
  );

  const description = useMemo(() => {
    return isAddedToRoot ? t('workflows.builder.removeFromForm') : t('workflows.builder.addToForm');
  }, [isAddedToRoot, t]);

  const icon = useMemo(() => {
    return isAddedToRoot ? <PiMinusBold /> : <PiPlusBold />;
  }, [isAddedToRoot]);

  const onClick = useCallback(() => {
    return isAddedToRoot ? removeNodeSettingFromRoot() : addNodeSettingToRoot();
  }, [isAddedToRoot, addNodeSettingToRoot, removeNodeSettingFromRoot]);

  return (
    <IconButton
      className={`${NO_DRAG_CLASS} node-setting-action-button`}
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

NodeSettingAddRemoveFormRoot.displayName = 'NodeSettingAddRemoveFormRoot';
