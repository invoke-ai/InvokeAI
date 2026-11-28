import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem } from '@features/gallery/contracts';
import { GALLERY_CULL_TOGGLE_COMMAND_ID, getGalleryCullMode, getGalleryCullModeValues } from '@features/gallery/cull';
import { ToggleIconButton } from '@platform/ui/Button';
import { useShortcutLabel } from '@workbench/hotkeys/useShortcutBinding';
import { AutoSwitchMenu } from '@workbench/widgets/shared/AutoSwitchMenu';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { ListChecksIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { useLivePreviewFollow } from './livePreviewFollow';
import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewDetailsPanel } from './PreviewDetailsPanel';
import { usePreviewHeaderContext, usePreviewStageContext, type PreviewZoomControls } from './previewHeaderStore';
import { getPreviewMetadataOpen } from './previewSettings';
import { PreviewZoomMenu } from './PreviewZoomMenu';

const Divider = () => <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />;

const AUTO_SWITCH_POSITIONING = { placement: 'bottom-end' } as const;

/** Selection actions, then zoom, auto-switch and Details, sized by the view's published density. */
export const PreviewHeaderActions = (_props: WidgetViewProps) => {
  const livePreview = useLivePreviewFollow();
  const autoSwitch = useActiveProjectSelector((project) => project.settings.viewerAutoSwitch);
  const {
    actionItem,
    actions,
    cancelCompare,
    copyCurrentVideoFrame,
    getRegion,
    isVideoFrameCopyAvailable,
    openItemMenu,
    position,
    zoom,
  } = usePreviewHeaderContext();
  const { density, stageElement, zoom: zoomReadout } = usePreviewStageContext();
  const zoomControls = useMemo<PreviewZoomControls | null>(
    () => (zoom && zoomReadout ? { ...zoom, ...zoomReadout } : null),
    [zoom, zoomReadout]
  );
  const isDetailsOpen = useActiveProjectSelector((project) =>
    getPreviewMetadataOpen(getProjectWidgetValues(project, 'preview'))
  );
  const { account, widgets } = useWorkbenchCommands();
  // Compact until the view has measured; minimal chrome is too narrow for more.
  const controlDensity = density === 'full' ? 'full' : 'compact';
  // A chosen mode follows the newest work again, so a session pinned from a tile lets go.
  const setAutoSwitch = useCallback(
    (mode: ViewerAutoSwitchMode) => {
      livePreview.showAll();
      account.updateProjectPreferences({ viewerAutoSwitch: mode });
    },
    [account, livePreview]
  );
  const setDetailsOpen = useCallback(
    (open: boolean) => widgets.patchValues('preview', { metadataOpen: open }),
    [widgets]
  );
  const hasItem = actionItem !== null && actions !== null;
  const { t } = useTranslation();
  const isCullModeOn = useActiveProjectSelector(
    (project) => getGalleryCullMode(getProjectWidgetValues(project, 'gallery')).enabled
  );
  const cullShortcut = useShortcutLabel(GALLERY_CULL_TOGGLE_COMMAND_ID);
  const cullLabel = t('widgets.gallery.cull.toggle');
  const setCullMode = useCallback(
    (enabled: boolean) => widgets.patchValues('gallery', getGalleryCullModeValues({ enabled })),
    [widgets]
  );

  return (
    <HStack gap="1">
      {hasItem ? (
        <>
          <PreviewActionStrip
            actions={actions}
            // The strip has a band of its own below compact; until the view measures, it is compact.
            density={density ?? 'compact'}
            getRegion={getRegion ?? undefined}
            isVideoFrameCopyAvailable={isVideoFrameCopyAvailable}
            item={actionItem}
            onCancelCompare={cancelCompare ?? undefined}
            onCopyCurrentFrame={copyCurrentVideoFrame ?? undefined}
            onOpenMenu={openItemMenu}
          />
          <Divider />
        </>
      ) : null}
      {/* Compact chrome has no room for a readout at fit, but a zoomed image must
          still say so and offer the way back. */}
      {zoomControls && (controlDensity === 'full' || zoomControls.isZoomed) ? (
        <PreviewZoomMenu density={controlDensity} zoom={zoomControls} />
      ) : null}
      <ToggleIconButton
        checked={isCullModeOn}
        data-preview-cull-toggle=""
        icon={ListChecksIcon}
        label={cullLabel}
        tooltip={
          cullShortcut
            ? t('widgets.gallery.cull.toggleTooltipWithShortcut', { shortcut: cullShortcut })
            : t('widgets.gallery.cull.toggleTooltip')
        }
        onCheckedChange={setCullMode}
      />
      <AutoSwitchMenu
        chrome="header"
        density="compact"
        positioning={AUTO_SWITCH_POSITIONING}
        value={autoSwitch}
        onChange={setAutoSwitch}
      />
      {hasItem ? (
        <PreviewDetailsPanel
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
