import type { QueueGenerationMeta, QueueReadModel } from '@features/queue/contracts';
import type { CanvasCandidateSlot } from '@workbench/canvasStagingView';

import { Menu, Portal } from '@chakra-ui/react';
import {
  buildProjectQueueItemOriginPrefix,
  buildQueueItemOrigin,
  extractGenerationMeta,
} from '@features/queue/contracts';
import { getQueueReadModelOptions } from '@features/queue/queries';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { useQuery } from '@tanstack/react-query';
import { QueueRecallMenuGroup } from '@workbench/queue-integration/QueueRecallMenuGroup';
import { useQueueItemRecall } from '@workbench/queue-integration/useQueueItemRecall';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { CheckIcon, SaveIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export interface StagingItemContextMenuTarget {
  slot: CanvasCandidateSlot;
  x: number;
  y: number;
}

/** A staged result carries no executed field values; the local submission snapshot is the source. */
const NO_META: QueueGenerationMeta = {};

/** Read executed prompts and seed from queue metadata; randomized seeds are unavailable from authored settings. */
const useStagedCandidateMeta = (projectId: string, backendItemId: number | undefined): QueueGenerationMeta => {
  const scope = useMemo(() => ({ originPrefix: buildProjectQueueItemOriginPrefix(projectId) }), [projectId]);
  const select = useCallback(
    (model: QueueReadModel): QueueGenerationMeta => {
      const item = model.items.find((candidate) => candidate.id === backendItemId);
      return item ? extractGenerationMeta(item) : NO_META;
    },
    [backendItemId]
  );
  return useQuery({ ...getQueueReadModelOptions(scope), enabled: backendItemId !== undefined, select }).data ?? NO_META;
};

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
  const projectId = useActiveProjectId();
  const meta = useStagedCandidateMeta(projectId, target.slot.candidate.sourceBackendItemId);
  const recall = useQueueItemRecall(buildQueueItemOrigin(target.slot.queueItemId, projectId), meta);
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
            <QueueRecallMenuGroup recall={recall} />
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
