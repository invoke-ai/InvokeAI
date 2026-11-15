import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';

import { Menu, Portal } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createMenuPartLoader, MenuActionItem, MenuContent, MenuPartPlaceholder, useMenuPart } from '@platform/ui/Menu';
import { CheckIcon, SaveIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export interface StagingItemContextMenuTarget {
  slot: CanvasCandidateSlot;
  x: number;
  y: number;
}

// The recall verbs and the queue item read behind them load on demand, off the Canvas's startup (the queue rows' and
// details' menus load them too). Pointing at or focusing a staged result warms them; the menu loads them as it opens.
const stagingItemRecall = createMenuPartLoader(() =>
  import('./StagingItemRecallGroup').then((module) => module.StagingItemRecallGroup)
);
export const loadStagingItemRecall = stagingItemRecall.load;

export const StagingItemContextMenu = ({
  canAccept,
  onAccept,
  onClose,
  onDiscard,
  onSaveToGallery,
  target,
}: {
  canAccept: boolean;
  onAccept: () => void;
  onClose: () => void;
  onDiscard: () => void;
  onSaveToGallery: () => void;
  target: StagingItemContextMenuTarget;
}) => {
  const { t } = useTranslation();
  const recall = useMenuPart(stagingItemRecall);
  // Mounted only while open, so mounting is the open: whoever opens the menu, its verbs load.
  useMountEffect(stagingItemRecall.load);
  const positioning = useMemo(
    () => ({
      getAnchorRect: () => ({ height: 1, width: 1, x: target.x, y: target.y }),
      placement: 'top-start' as const,
    }),
    [target.x, target.y]
  );
  const onOpenChange = useCallback(
    (details: { open: boolean }) => {
      if (!details.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Menu.Root lazyMount open positioning={positioning} unmountOnExit onOpenChange={onOpenChange}>
      <Portal>
        <Menu.Positioner>
          <MenuContent aria-busy={recall.status === 'loading'} minW="13rem" py="1">
            {/* The menu opens at once, so its other items, Escape and an outside press work while the verbs load. */}
            {recall.status === 'loaded' ? (
              <recall.part slot={target.slot} />
            ) : (
              <MenuPartPlaceholder status={recall.status} />
            )}
            <Menu.Separator borderColor="border.subtle" />
            <MenuActionItem
              icon={SaveIcon}
              label={t('widgets.canvas.staging.saveToGallery')}
              value="save"
              onSelect={onSaveToGallery}
            />
            <MenuActionItem
              disabled={!canAccept}
              icon={CheckIcon}
              label={t('widgets.canvas.acceptToLayer')}
              value="accept"
              onSelect={onAccept}
            />
            <MenuActionItem icon={XIcon} label={t('common.discard')} value="discard" onSelect={onDiscard} />
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
