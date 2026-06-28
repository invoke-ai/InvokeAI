import { SegmentGroup } from '@chakra-ui/react';
import { useCallback } from 'react';

import type { QueueFilterId } from './queueFilters';

import { QUEUE_FILTERS } from './queueFilters';

/** Status filter pills for the RECENT list (All · Active · Done · Failed · Canceled). */
export const QueueFilterTabs = ({
  value,
  onChange,
}: {
  value: QueueFilterId;
  onChange: (filter: QueueFilterId) => void;
}) => {
  const onValueChange = useCallback(
    (details: { value: string | null }) => {
      if (details.value) {
        onChange(details.value as QueueFilterId);
      }
    },
    [onChange]
  );

  return (
    <SegmentGroup.Root
      aria-label="Filter recent queue items by status"
      colorPalette="accent"
      size="xs"
      value={value}
      onValueChange={onValueChange}
    >
      <SegmentGroup.Indicator />
      {QUEUE_FILTERS.map((filter) => (
        <SegmentGroup.Item key={filter.id} value={filter.id}>
          <SegmentGroup.ItemHiddenInput />
          <SegmentGroup.ItemText>{filter.label}</SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};
