import { MenuItem } from '@invoke-ai/ui-library';
import { useBuildWorkflowFast } from 'features/nodes/util/workflow/buildWorkflow';
import { saveWorkflowAs } from 'features/workflowLibrary/components/SaveWorkflowAsDialog';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

const SaveWorkflowAsMenuItem = () => {
  const { t } = useTranslation();
  const buildWorkflow = useBuildWorkflowFast();

  const handleClickSave = useCallback<MouseEventHandler<HTMLButtonElement>>(
    (e) => {
      e.stopPropagation();
      const workflow = buildWorkflow();
      saveWorkflowAs(workflow);
    },
    [buildWorkflow]
  );

  return (
    <MenuItem as="button" icon={<PiCopyBold />} onClick={handleClickSave}>
      {t('workflows.saveWorkflowAs')}
    </MenuItem>
  );
};

export default memo(SaveWorkflowAsMenuItem);
