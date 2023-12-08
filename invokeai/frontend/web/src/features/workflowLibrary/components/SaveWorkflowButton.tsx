import IAIIconButton from 'common/components/IAIIconButton';
import { useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSave } from 'react-icons/fa';

const SaveLibraryWorkflowButton = () => {
  const { t } = useTranslation();
  const { saveWorkflow, isLoading } = useSaveLibraryWorkflow();
  return (
    <IAIIconButton
      icon={<FaSave />}
      onClick={saveWorkflow}
      isLoading={isLoading}
      tooltip={t('workflows.saveWorkflow')}
      aria-label={t('workflows.saveWorkflow')}
    />
  );
};

export default memo(SaveLibraryWorkflowButton);
