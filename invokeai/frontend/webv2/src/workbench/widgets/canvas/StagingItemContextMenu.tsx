import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';

import { Menu, Portal } from '@chakra-ui/react';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { CheckIcon, SaveIcon, XIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export interface StagingItemContextMenuTarget {
  slot: CanvasCandidateSlot;
  x: number;
  y: number;
}

// The recall verbs and the queue item read behind them load with the menu, off the Canvas's startup: the queue rows'
// and details' menus share them lazily too.
const StagingItemRecallGroup = lazy(() =>
  import('./StagingItemRecallGroup').then((module) => ({ default: module.StagingItemRecallGroup }))
);

const ignoreSelect = () => undefined;

/** A disabled entry in the recall verbs' place, so the menu holds only menu items while they load. */
const RecallEntriesLoading = () => {
  const { t } = useTranslation();

  return <MenuActionItem disabled label={t('common.loading')} value="loading" onSelect={ignoreSelect} />;
};
const RECALL_ENTRIES_FALLBACK = <RecallEntriesLoading />;

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
          <MenuContent minW="13rem" py="1">
            {/* The menu opens at once, so its other items, Escape and an outside press work while the verbs load. */}
            <Suspense fallback={RECALL_ENTRIES_FALLBACK}>
              <StagingItemRecallGroup slot={target.slot} />
            </Suspense>
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
