import type { ViewerAutoSwitchMode } from '@workbench/settings/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, HStack } from '@chakra-ui/react';
import { galleryImageItemToGalleryImage, isGalleryImageItem } from '@features/gallery/contracts';
import { AutoSwitchMenu } from '@workbench/widgets/shared/AutoSwitchMenu';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

import { useLivePreviewFollow } from './livePreviewFollow';
import { PreviewActionStrip } from './PreviewActionStrip';
import { PreviewDetailsPanel } from './PreviewDetailsPanel';
import { usePreviewHeaderContext, usePreviewStageContext, type PreviewZoomControls } from './previewHeaderStore';
import { getPreviewMetadataOpen } from './previewSettings';
import { PreviewZoomMenu } from './PreviewZoomMenu';

const Divider = () => <Box bg="border.subtle" flexShrink={0} h="4" w="1px" />;

const AUTO_SWITCH_POSITIONING = { placement: 'bottom-end' } as const;

/**
 * The selection's actions, then the viewer's own controls (zoom, auto-switch, Details), sized by the density the view
 * publishes: its measured width in the center or a floating window (side rails are always compact). The filmstrip toggle lives in the gear's quick settings and on the `T` key.
 */
export const PreviewHeaderActions = (_props: WidgetViewProps) => {
  const livePreview = useLivePreviewFollow();
  const autoSwitch = useActiveProjectSelector((project) => project.settings.viewerAutoSwitch);
  const { actionItem, actions, copyCurrentVideoFrame, isVideoFrameCopyAvailable, openItemMenu, position, zoom } =
    usePreviewHeaderContext();
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

  return (
    <HStack gap="1">
      {hasItem ? (
        <>
          <PreviewActionStrip
            actions={actions}
            // The strip has a band of its own below compact; until the view measures, it is compact.
            density={density ?? 'compact'}
            isVideoFrameCopyAvailable={isVideoFrameCopyAvailable}
            item={actionItem}
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
