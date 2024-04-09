import { MenuItem } from '@invoke-ai/ui-library';
import { useLoadWorkflowFromGraphModal } from 'features/workflowLibrary/components/LoadWorkflowFromGraphModal/LoadWorkflowFromGraphModal';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlaskBold } from 'react-icons/pi';

const LoadWorkflowFromGraphMenuItem = () => {
  const { t } = useTranslation();
  const { onOpen } = useLoadWorkflowFromGraphModal();

  return (
    <MenuItem as="button" icon={<PiFlaskBold />} onClick={onOpen}>
      {t('workflows.loadFromGraph')}
    </MenuItem>
  );
};

export default memo(LoadWorkflowFromGraphMenuItem);
