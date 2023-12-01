import IAIIconButton from 'common/components/IAIIconButton';
import { useDuplicateLibraryWorkflow } from 'features/workflowLibrary/hooks/useDuplicateWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaClone } from 'react-icons/fa';

const DuplicateLibraryWorkflowButton = () => {
  const { t } = useTranslation();
  const { duplicateWorkflow, isLoading } = useDuplicateLibraryWorkflow();
  return (
    <IAIIconButton
      icon={<FaClone />}
      onClick={duplicateWorkflow}
      isLoading={isLoading}
      tooltip={t('workflows.duplicateWorkflow')}
      aria-label={t('workflows.duplicateWorkflow')}
    />
  );
};

export default memo(DuplicateLibraryWorkflowButton);
