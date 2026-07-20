import type { SystemStyleObject } from '@chakra-ui/react';
import type { QueueItemReadModel } from '@features/queue/core/types';

import { Box, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { extractGenerationMeta, getResultImageName } from '@features/queue/core/generationMeta';
import { useItemProgress } from '@features/queue/data/itemProgressStore';
import { Row } from '@platform/ui/Row';
import { ChevronRightIcon } from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { CancelQueueItemButton } from './CancelQueueItemButton';
import { formatCompactAge, formatDuration } from './formatDuration';
import { QueueItemDetails } from './QueueItemDetails';
import { QueueItemThumbnail } from './QueueItemThumbnail';
import { QueueStatusDot } from './QueueStatusDot';
import { QueueStepProgress } from './QueueStepProgress';
import { useQueueUi } from './QueueUiContext';
import { getStatusMeta } from './statusMeta';

const CHEVRON_OPEN = { transform: 'rotate(90deg)' } as const;

const QUEUE_ITEM_BUTTON_SX: SystemStyleObject = {
  display: 'flex',
  alignItems: 'center',
  flex: 1,
  gap: 2.5,
  minH: 11,
  minW: 0,
  px: 2,
  py: 1.5,
  textAlign: 'start',
  rounded: 'none',
  transitionDuration: 'fastest',
} as const;

export const QueueItemRow = memo(({ item }: { item: QueueItemReadModel }) => {
  const { t } = useTranslation();
  const { canManageItem, canViewItemDetails } = useQueueUi();
  const [expanded, setExpanded] = useState(false);
  const toggle = useCallback(() => setExpanded((open) => !open), []);
  const meta = extractGenerationMeta(item);
  const duration = formatDuration(item.startedAt, item.completedAt);
  const age = formatCompactAge(item.completedAt ?? item.createdAt);
  const ageLabel = [duration, age].filter(Boolean).join(' · ');
  const isFailed = item.status === 'failed';
  const isCancellable = (item.status === 'pending' || item.status === 'in_progress') && canManageItem(item);
  const canExpand = canViewItemDetails(item);
  const progress = useItemProgress(item.id);
  const liveImage = progress?.image ?? null;
  const resultImageName = getResultImageName(item);
  const statusLabel = t(getStatusMeta(item.status).labelKey);

  const showBorder = expanded || isFailed;
  const borderColor = showBorder ? (isFailed ? 'fg.error' : 'border') : 'transparent';

  return (
    <Box overflow="hidden" rounded="md" borderWidth={1} borderColor={borderColor}>
      <HStack gap="0">
        <Row
          aria-expanded={canExpand ? expanded : undefined}
          as={canExpand ? 'button' : 'div'}
          css={QUEUE_ITEM_BUTTON_SX}
          onClick={canExpand ? toggle : undefined}
        >
          <QueueItemThumbnail boxSize="8" imageName={resultImageName} liveImage={liveImage} />
          <Stack flex="1" gap="0.5" minW="0">
            <Text fontSize="xs" truncate>
              {meta.positivePrompt?.trim() || t('widgets.queue.noPrompt')}
            </Text>
            <HStack gap="1.5" minW="0">
              <QueueStatusDot status={item.status} />
              <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums" truncate>
                {[statusLabel, ageLabel].filter(Boolean).join(' · ')}
              </Text>
            </HStack>
          </Stack>
          {canExpand ? (
            <Icon
              as={ChevronRightIcon}
              boxSize="4"
              color="fg.muted"
              flexShrink={0}
              transition="transform var(--wb-motion-duration-fast) ease"
              css={expanded ? CHEVRON_OPEN : undefined}
            />
          ) : null}
        </Row>
        {isCancellable ? (
          <Box flexShrink={0} pe="1">
            <CancelQueueItemButton itemId={item.id} />
          </Box>
        ) : null}
      </HStack>

      {item.status === 'in_progress' ? (
        <Box px="2.5" pb="2">
          <QueueStepProgress message={progress?.message ?? ''} percentage={progress?.percentage ?? null} />
        </Box>
      ) : null}

      {expanded ? (
        <Box pb="2.5" pt="1" px="2.5">
          <QueueItemDetails item={item} />
        </Box>
      ) : null}
    </Box>
  );
});
