import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem } from '@features/gallery/contracts';
import { ToggleIconButton } from '@platform/ui';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GalleryThumbnailsIcon, HourglassIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useLivePreviewFollow } from './livePreviewFollow';
import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewDetailsPopover } from './PreviewDetailsPopover';
import { usePreviewHeaderContext, usePreviewStageContext, type PreviewZoomControls } from './previewHeaderStore';
import { getPreviewFilmstripVisible, getPreviewMetadataOpen } from './previewSettings';
import { PreviewZoomMenu } from './PreviewZoomMenu';

const Divider = () => <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />;

export const PreviewHeaderActions = ({ region }: WidgetViewProps) => {
  const { t } = useTranslation();
  const livePreview = useLivePreviewFollow();
  const showProgressImagesInViewer = useActiveProjectSelector((project) => project.settings.showProgressImagesInViewer);
  const hasProgressImage = livePreview.sessions.length > 0;
  const { actionItem, actions, copyCurrentVideoFrame, isVideoFrameCopyAvailable, openItemMenu, position, zoom } =
    usePreviewHeaderContext();
  const { stageElement, zoom: zoomReadout } = usePreviewStageContext();
  const zoomControls = useMemo<PreviewZoomControls | null>(
    () => (zoom && zoomReadout ? { ...zoom, ...zoomReadout } : null),
    [zoom, zoomReadout]
  );
  const { isDetailsOpen, isFilmstripVisible } = useActiveProjectSelector((project) => {
    const values = getProjectWidgetValues(project, 'preview');

    return { isDetailsOpen: getPreviewMetadataOpen(values), isFilmstripVisible: getPreviewFilmstripVisible(values) };
  });
  const { account, widgets } = useWorkbenchCommands();
  const isFull = region === 'center';
  const label = showProgressImagesInViewer
    ? t('widgets.preview.hideInProgressDiffusion')
    : t('widgets.preview.showInProgressDiffusion');
  const filmstripLabel = isFilmstripVisible ? t('widgets.preview.hideFilmstrip') : t('widgets.preview.showFilmstrip');
  const toggleProgressImages = useCallback(() => {
    livePreview.showAll();
    account.updateProjectPreferences({ showProgressImagesInViewer: !showProgressImagesInViewer });
  }, [account, livePreview, showProgressImagesInViewer]);
  const toggleFilmstrip = useCallback(
    () => widgets.patchValues('preview', { filmstripVisible: !isFilmstripVisible }),
    [isFilmstripVisible, widgets]
  );
  const setDetailsOpen = useCallback(
    (open: boolean) => widgets.patchValues('preview', { metadataOpen: open }),
    [widgets]
  );

  return (
    <HStack gap="1">
      {/* Compact chrome has no room for a readout at fit, but a zoomed image must
          still say so and offer the way back. */}
      {zoomControls && (isFull || zoomControls.isZoomed) ? (
        <>
          <PreviewZoomMenu zoom={zoomControls} />
          <Divider />
        </>
      ) : null}
      {actionItem && actions ? (
        <>
          <PreviewActionStrip
            actions={actions}
            density={isFull ? 'full' : 'compact'}
            isVideoFrameCopyAvailable={isVideoFrameCopyAvailable}
            item={actionItem}
            onCopyCurrentFrame={copyCurrentVideoFrame ?? undefined}
            onOpenMenu={openItemMenu}
          />
          <Divider />
        </>
      ) : null}

      <ToggleIconButton
        checked={showProgressImagesInViewer}
        icon={HourglassIcon}
        label={label}
        // Apply dimming after primitive props so idle-state styling survives.
        opacity={hasProgressImage || showProgressImagesInViewer ? 1 : 0.7}
        onCheckedChange={toggleProgressImages}
      />
      <ToggleIconButton
        checked={isFilmstripVisible}
        icon={GalleryThumbnailsIcon}
        label={filmstripLabel}
        onCheckedChange={toggleFilmstrip}
      />
      {actionItem && actions ? (
        <PreviewDetailsPopover
          actions={actions}
          image={isGalleryImageItem(actionItem) ? galleryImageItemToGalleryImage(actionItem) : null}
          isOpen={isDetailsOpen}
          item={actionItem}
          position={position}
          stageElement={stageElement}
          onOpenChange={setDetailsOpen}
        />
      ) : null}
    </HStack>
  );
};
