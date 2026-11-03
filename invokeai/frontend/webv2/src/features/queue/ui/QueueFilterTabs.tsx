import { Text } from '@chakra-ui/react';
import { SegmentTabs, type SegmentTab } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { useQueueCounts } from './queueDataStore';
import { getFilterCount, QUEUE_FILTERS } from './queueFilters';

export const QUEUE_FILTER_TABS_ID = 'queue-filter';

/**
 * The queue's counts are its filters: each status tab carries how many items the scope holds in it, failures tinted
 * while there are any (All · Active · Done · Failed · Canceled).
 */
export const QueueFilterTabs = ({
  value,
  onChange,
}: {
  value: QueueFilterId;
  onChange: (filter: QueueFilterId) => void;
}) => {
  const { i18n, t } = useTranslation();
  const counts = useQueueCounts();
  const formatCount = useMemo(() => new Intl.NumberFormat(i18n.resolvedLanguage).format, [i18n.resolvedLanguage]);
  const tabs = useMemo<SegmentTab<QueueFilterId>[]>(
    () =>
      QUEUE_FILTERS.map((filter) => {
        const count = getFilterCount(counts, filter.id);

        return {
          id: filter.id,
          label: (
            <Text alignItems="center" as="span" display="flex" gap="1.5">
              {t(filter.labelKey)}
              <Text
                as="span"
                color={filter.id === 'failed' && count > 0 ? 'fg.error' : 'currentColor'}
                fontVariantNumeric="tabular-nums"
                opacity={filter.id === 'failed' && count > 0 ? 1 : 0.8}
              >
                {formatCount(count)}
              </Text>
            </Text>
          ),
        };
      }),
    [counts, formatCount, t]
  );

  return (
    <SegmentTabs
      isCompact
      activeId={value}
      ariaLabel={t('widgets.queue.filterRecentByStatus')}
      idBase={QUEUE_FILTER_TABS_ID}
      tabs={tabs}
      onSelect={onChange}
    />
  );
};
