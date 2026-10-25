import type { CanvasDocumentContractV3, CanvasEngineLayerCapability } from '@workbench/canvas-engine/api';
import type { Notify } from '@workbench/useNotify';
import type { TFunction } from 'i18next';

import { getDocumentIndex } from '@workbench/canvas-engine/api';

import { reportLayerOperation } from './useStructuralCommit';

export interface CanvasContextMenuTarget {
  x: number;
  y: number;
  layerId: string | null;
}

interface ResolveCanvasContextMenuOptions {
  clientX: number;
  clientY: number;
  /** Target the selected layer without hit-testing right-click pixels; the layers panel owns selection. */
  selectedLayerId?: string | null;
  isInlineEditor: boolean;
  isInteractionLocked: boolean;
}

interface CanvasContextMenuResolution {
  preventDefault: boolean;
  target: CanvasContextMenuTarget | null;
}

export type CanvasContextMenuBranch = 'global' | 'layer' | null;

export const resolveCanvasContextMenuBranch = (
  target: CanvasContextMenuTarget | null,
  hasEngine: boolean
): CanvasContextMenuBranch => {
  if (!target) {
    return null;
  }

  return target.layerId !== null && hasEngine ? 'layer' : 'global';
};

export const resolveCanvasContextMenu = ({
  clientX,
  clientY,
  isInlineEditor,
  isInteractionLocked,
  selectedLayerId,
}: ResolveCanvasContextMenuOptions): CanvasContextMenuResolution => {
  if (isInlineEditor) {
    return { preventDefault: false, target: null };
  }

  // A locked surface offers only the global menu — there is nothing to act on.
  const layerId = isInteractionLocked ? null : (selectedLayerId ?? null);

  return {
    preventDefault: true,
    target: { layerId, x: clientX, y: clientY },
  };
};

/**
 * Why Crop all layers to frame cannot run on `document`, or null when it has a leaf to act on: one not locked by
 * itself or a group above it.
 */
export const getCropAllUnavailableReasonKey = (document: CanvasDocumentContractV3): string | null => {
  const { byId, leaves } = getDocumentIndex(document);
  if (leaves.length === 0) {
    return 'widgets.canvas.contextMenu.cropAllNoLayers';
  }
  return leaves.some((leaf) => !leaf.isLocked && !byId.get(leaf.id)?.ancestorsLocked)
    ? null
    : 'widgets.canvas.contextMenu.cropAllAllLocked';
};

/**
 * Crops every unlocked layer to the frame as one undo step and says what happened: how many layers it cropped, how
 * many locked ones it skipped, which layer could not be read (named, with a count of any others), or that nothing
 * needed cropping. The canvas menu and the command share it.
 */
export const cropAllLayersToFrame = async (
  layers: Pick<CanvasEngineLayerCapability, 'cropAllLayersToBbox'>,
  notify: Pick<Notify, 'error' | 'info' | 'success'>,
  t: TFunction
): Promise<void> => {
  let result: Awaited<ReturnType<typeof layers.cropAllLayersToBbox>>;
  try {
    result = await layers.cropAllLayersToBbox();
  } catch {
    // The engine leaves every layer as it was when the transaction is rejected.
    result = { status: 'failed' };
  }
  switch (result.status) {
    case 'cropped':
    case 'nothing': {
      const { skippedLocked, skippedUnreadable } = result;
      const [firstUnreadable] = skippedUnreadable;
      const skipped =
        [
          skippedLocked > 0 ? t('widgets.canvas.contextMenu.croppedAllLayersSkipped', { count: skippedLocked }) : '',
          !firstUnreadable
            ? ''
            : skippedUnreadable.length === 1
              ? t('widgets.canvas.contextMenu.cropAllUnreadable', { name: firstUnreadable.layerName })
              : t('widgets.canvas.contextMenu.cropAllUnreadableWithOthers', {
                  count: skippedUnreadable.length - 1,
                  name: firstUnreadable.layerName,
                }),
        ]
          .filter(Boolean)
          .join(' ') || undefined;
      if (result.status === 'cropped') {
        notify.success(t('widgets.canvas.contextMenu.croppedAllLayers', { count: result.cropped }), skipped);
      } else if (firstUnreadable) {
        // Those layers needed cropping and could not be read: nothing the user asked for happened.
        notify.error(t('widgets.canvas.contextMenu.cropAllNoneCropped'), skipped);
      } else {
        notify.info(
          t(
            skipped ? 'widgets.canvas.contextMenu.cropAllNothingUnlocked' : 'widgets.canvas.contextMenu.cropAllNothing'
          ),
          skipped
        );
      }
      return;
    }
    case 'adjustment-refused':
      reportLayerOperation(result, notify.error, t);
      return;
    default:
      reportLayerOperation(result.status, notify.error, t);
  }
};
