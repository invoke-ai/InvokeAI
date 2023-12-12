import { MenuItem } from '@chakra-ui/react';
import { useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSave } from 'react-icons/fa';

const SaveLibraryWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const { saveWorkflow } = useSaveLibraryWorkflow();
  return (
    <MenuItem as="button" icon={<FaSave />} onClick={saveWorkflow}>
      {t('workflows.saveWorkflow')}
    </MenuItem>
  );
};

export default memo(SaveLibraryWorkflowMenuItem);
