import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem } from '@features/gallery/contracts';
import { ToggleIconButton } from '@platform/ui';
import { AutoSwitchMenu } from '@workbench/widgets/shared/AutoSwitchMenu';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GalleryThumbnailsIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useLivePreviewFollow } from './livePreviewFollow';
import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewDetailsPopover } from './PreviewDetailsPopover';
import { usePreviewHeaderContext, usePreviewStageContext, type PreviewZoomControls } from './previewHeaderStore';
import { getPreviewFilmstripVisible, getPreviewMetadataOpen } from './previewSettings';
import { PreviewZoomMenu } from './PreviewZoomMenu';

const Divider = () => <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />;

const AUTO_SWITCH_POSITIONING = { placement: 'bottom-end' } as const;

export const PreviewHeaderActions = ({ region }: WidgetViewProps) => {
  const { t } = useTranslation();
  const livePreview = useLivePreviewFollow();
  const autoSwitch = useActiveProjectSelector((project) => project.settings.viewerAutoSwitch);
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
  const filmstripLabel = isFilmstripVisible ? t('widgets.preview.hideFilmstrip') : t('widgets.preview.showFilmstrip');
  // A chosen mode follows the newest work again, so a session pinned from a tile lets go.
  const setAutoSwitch = useCallback(
    (mode: ViewerAutoSwitchMode) => {
      livePreview.showAll();
      account.updateProjectPreferences({ viewerAutoSwitch: mode });
    },
    [account, livePreview]
  );
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

      <AutoSwitchMenu
        density={isFull ? 'full' : 'compact'}
        positioning={AUTO_SWITCH_POSITIONING}
        value={autoSwitch}
        onChange={setAutoSwitch}
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
