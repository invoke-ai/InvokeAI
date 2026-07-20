import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack, Icon } from '@chakra-ui/react';
import { useProgressImage } from '@features/queue/react';
import { IconButton } from '@platform/ui';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GalleryThumbnailsIcon, HourglassIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { PreviewActionStrip } from './PreviewActionStrip';
import { usePreviewHeaderContext } from './previewHeaderStore';
import { getPreviewFilmstripVisible } from './previewSettings';

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
  const { actionImage, actions, openImageMenu } = usePreviewHeaderContext();
  const isFilmstripVisible = useActiveProjectSelector((project) =>
    getPreviewFilmstripVisible(getProjectWidgetValues(project, 'preview'))
  );
  const { account, widgets } = useWorkbenchCommands();
  const label = showProgressImagesInViewer
    ? t('widgets.preview.hideInProgressDiffusion')
    : t('widgets.preview.showInProgressDiffusion');
  const filmstripLabel = isFilmstripVisible ? t('widgets.preview.hideFilmstrip') : t('widgets.preview.showFilmstrip');
  const toggleProgressImages = useCallback(
    () => account.updateProjectPreferences({ showProgressImagesInViewer: !showProgressImagesInViewer }),
    [account, showProgressImagesInViewer]
  );
  const toggleFilmstrip = useCallback(
    () => widgets.patchValues('preview', { filmstripVisible: !isFilmstripVisible }),
    [isFilmstripVisible, widgets]
  );

  return (
    <HStack gap="1">
      {actionImage && actions ? (
        <>
          <PreviewActionStrip
            actions={actions}
            density={region === 'center' ? 'full' : 'compact'}
            image={actionImage}
            onOpenMenu={openImageMenu}
          />
          <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />
        </>
      ) : null}
      <IconButton
        aria-label={filmstripLabel}
        color={isFilmstripVisible ? 'fg' : 'fg.muted'}
        size="2xs"
        title={filmstripLabel}
        variant="ghost"
        onClick={toggleFilmstrip}
      >
        <Icon as={GalleryThumbnailsIcon} boxSize="3.5" />
      </IconButton>
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
