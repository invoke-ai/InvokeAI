import { MenuItem } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { SpinnerIcon } from 'features/gallery/components/ImageContextMenu/SpinnerIcon';
import { useImageDTOContext } from 'features/gallery/contexts/ImageDTOContext';
import { $hasTemplates } from 'features/nodes/store/nodesSlice';
import { useGetAndLoadEmbeddedWorkflow } from 'features/workflowLibrary/hooks/useGetAndLoadEmbeddedWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';

export const ImageMenuItemLoadWorkflow = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useImageDTOContext();
  const [getAndLoadEmbeddedWorkflow, { isLoading }] = useGetAndLoadEmbeddedWorkflow();
  const hasTemplates = useStore($hasTemplates);

  const onClick = useCallback(() => {
    getAndLoadEmbeddedWorkflow(imageDTO.image_name);
  }, [getAndLoadEmbeddedWorkflow, imageDTO.image_name]);

  return (
    <MenuItem
      icon={isLoading ? <SpinnerIcon /> : <PiFlowArrowBold />}
      onClickCapture={onClick}
      isDisabled={!imageDTO.has_workflow || !hasTemplates}
    >
      {t('nodes.loadWorkflow')}
    </MenuItem>
  );
});

ImageMenuItemLoadWorkflow.displayName = 'ImageMenuItemLoadWorkflow';
