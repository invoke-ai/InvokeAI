import type { ImageRecallKind } from '@workbench/image-actions';

import { Menu } from '@chakra-ui/react';
import { MenuActionItem } from '@platform/ui/Menu';
import { getImageRecallVerb, IMAGE_RECALL_KINDS } from '@workbench/image-actions';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueItemRecall } from './useQueueItemRecall';

/**
 * A queued run's recall verbs, for every menu that offers them (staged results, queue rows, queue details). A kind it
 * cannot recall stays listed, unavailable with its reason.
 */
export const QueueRecallMenuGroup = ({ recall }: { recall: QueueItemRecall }) => {
  const { t } = useTranslation();

  return (
    <Menu.ItemGroup>
      <Menu.ItemGroupLabel>{t('widgets.canvas.staging.recall')}</Menu.ItemGroupLabel>
      {IMAGE_RECALL_KINDS.map((kind) => (
        <RecallItem key={kind} kind={kind} recall={recall} />
      ))}
    </Menu.ItemGroup>
  );
};

const RecallItem = ({ kind, recall }: { kind: ImageRecallKind; recall: QueueItemRecall }) => {
  const { t } = useTranslation();
  const verb = getImageRecallVerb(kind, t);
  const reason = recall.reasons[kind];
  const onSelect = useCallback(() => void recall.recall(kind), [kind, recall]);

  return (
    <MenuActionItem
      disabled={!recall.capabilities[kind]}
      icon={verb.icon}
      label={verb.label}
      unavailableReason={reason ? t(reason) : undefined}
      value={kind}
      onSelect={onSelect}
    />
  );
};
