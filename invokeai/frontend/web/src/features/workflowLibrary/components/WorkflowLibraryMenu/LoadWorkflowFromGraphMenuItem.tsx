import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $templates } from 'features/nodes/store/nodesSlice';
import { useLoadWorkflowFromGraphModal } from 'features/workflowLibrary/components/LoadWorkflowFromGraphModal/LoadWorkflowFromGraphModal';
import { size } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlaskBold } from 'react-icons/pi';

const LoadWorkflowFromGraphMenuItem = () => {
  const { t } = useTranslation();
  const templates = useStore($templates);
  const { onOpen } = useLoadWorkflowFromGraphModal();

  return (
    <MenuItem as="button" icon={<PiFlaskBold />} onClick={onOpen} isDisabled={!size(templates)}>
      {t('workflows.loadFromGraph')}
    </MenuItem>
  );
};

export default memo(LoadWorkflowFromGraphMenuItem);
