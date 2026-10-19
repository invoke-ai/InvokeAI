import { SegmentTabs, type SegmentTab } from '@platform/ui';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { QUEUE_FILTERS } from './queueFilters';

export const QUEUE_FILTER_TABS_ID = 'queue-filter';

/** Status filter tabs for the RECENT list (All · Active · Done · Failed · Canceled). */
export const QueueFilterTabs = ({
  value,
  onChange,
}: {
  value: QueueFilterId;
  onChange: (filter: QueueFilterId) => void;
}) => {
  const { t } = useTranslation();
  const tabs = useMemo<SegmentTab<QueueFilterId>[]>(
    () => QUEUE_FILTERS.map((filter) => ({ id: filter.id, label: t(filter.labelKey) })),
    [t]
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
