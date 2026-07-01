import { Icon } from '@chakra-ui/react';
import { useProgressImage } from '@workbench/backend/progressImageStore';
import { IconButton } from '@workbench/components/ui';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { HourglassIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const PreviewHeaderActions = () => {
  const { t } = useTranslation();
  const showProgressImagesInViewer = useActiveProjectSelector((project) => project.settings.showProgressImagesInViewer);
  const hasProgressImage = useProgressImage() !== null;
  const dispatch = useWorkbenchDispatch();
  const label = showProgressImagesInViewer
    ? t('widgets.preview.hideInProgressDiffusion')
    : t('widgets.preview.showInProgressDiffusion');
  const toggleProgressImages = useCallback(
    () =>
      dispatch({
        settings: { showProgressImagesInViewer: !showProgressImagesInViewer },
        type: 'setActiveProjectSettings',
      }),
    [dispatch, showProgressImagesInViewer]
  );

  return (
    <IconButton
      aria-label={label}
      colorPalette={showProgressImagesInViewer ? 'accent' : 'gray'}
      opacity={hasProgressImage || showProgressImagesInViewer ? 1 : 0.7}
      size="2xs"
      title={label}
      variant={showProgressImagesInViewer ? 'solid' : 'ghost'}
      onClick={toggleProgressImages}
    >
      <Icon as={HourglassIcon} boxSize="3.5" />
    </IconButton>
  );
};
