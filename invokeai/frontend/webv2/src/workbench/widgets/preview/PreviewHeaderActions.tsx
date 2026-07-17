import type { WidgetViewProps } from '@workbench/types';

import { Box, HStack, Icon } from '@chakra-ui/react';
import { useProgressImage } from '@workbench/backend/progressImageStore';
import { IconButton } from '@workbench/components/ui';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { HourglassIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewActionStrip } from './PreviewActionStrip';
import { usePreviewHeaderContext } from './previewHeaderStore';

/**
 * The preview widget's header actions: the image action strip for the current
 * selection (published by the view via `previewHeaderStore`) plus the
 * in-progress diffusion toggle. Actions live here — in the frame's standard
 * header slot — like every other widget, not in the widget body.
 */
export const PreviewHeaderActions = ({ region }: WidgetViewProps) => {
  const { t } = useTranslation();
  const showProgressImagesInViewer = useActiveProjectSelector((project) => project.settings.showProgressImagesInViewer);
  const hasProgressImage = useProgressImage() !== null;
  const { actionImage, actions } = usePreviewHeaderContext();
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
    <HStack gap="1">
      {actionImage && actions ? (
        <>
          <PreviewActionStrip
            actions={actions}
            density={region === 'center' ? 'full' : 'compact'}
            image={actionImage}
          />
          <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />
        </>
      ) : null}
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
    </HStack>
  );
};
