import type { ResultDestination } from '@workbench/invocationContracts';

import { SegmentGroup } from '@chakra-ui/react';
import { WidgetIcon } from '@workbench/iconResolver';
import { getDestinationLabel, resultDestinations } from '@workbench/invocation';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useCallback } from 'react';

const destinationWidgetTypeIds: Record<ResultDestination, 'canvas' | 'gallery'> = {
  canvas: 'canvas',
  gallery: 'gallery',
};

export const RoutingDestinationSegments = ({
  ariaLabel,
  disabled,
  onChange,
  value,
}: {
  ariaLabel: string;
  disabled?: boolean;
  onChange: (destination: ResultDestination) => void;
  value: ResultDestination | null;
}) => {
  const handleChange = useCallback(
    (event: { value: string | null }) => {
      if (event.value) {
        onChange(event.value as ResultDestination);
      }
    },
    [onChange]
  );

  return (
    <SegmentGroup.Root aria-label={ariaLabel} disabled={disabled} size="xs" value={value} onValueChange={handleChange}>
      <SegmentGroup.Indicator />
      {resultDestinations.map((destination) => (
        <SegmentGroup.Item key={destination.id} flex="1" justifyContent="center" value={destination.id}>
          <SegmentGroup.ItemHiddenInput />
          <SegmentGroup.ItemText display="flex" alignItems="center" gap="1.5">
            <WidgetIcon boxSize="3.5" icon={getWidgetById(destinationWidgetTypeIds[destination.id])?.manifest.icon} />
            {getDestinationLabel(destination.id)}
          </SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};
