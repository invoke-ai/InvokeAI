import type { ResultDestination } from '@workbench/invocationContracts';

import { SegmentGroup } from '@chakra-ui/react';
import { WidgetIcon } from '@workbench/iconResolver';
import { getDestinationLabel, resultDestinations } from '@workbench/invocation';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useCallback } from 'react';

export const RoutingDestinationSegments = ({
  ariaLabel,
  disabled,
  onChange,
  size,
  value,
}: {
  ariaLabel: string;
  disabled?: boolean;
  onChange: (destination: ResultDestination) => void;
  /** `xs` fits the dense Invoke card; unset keeps the dialog's working size. */
  size?: 'xs';
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
    <SegmentGroup.Root
      aria-label={ariaLabel}
      disabled={disabled}
      size={size}
      value={value}
      onValueChange={handleChange}
    >
      <SegmentGroup.Indicator />
      {resultDestinations.map((destination) => (
        <SegmentGroup.Item key={destination.id} flex="1" justifyContent="center" value={destination.id}>
          <SegmentGroup.ItemHiddenInput />
          <SegmentGroup.ItemText alignItems="center" display="flex" fontSize={size} gap={size === 'xs' ? '1' : '1.5'}>
            <WidgetIcon boxSize={size === 'xs' ? '3' : '3.5'} icon={getWidgetById(destination.id)?.manifest.icon} />
            {getDestinationLabel(destination.id)}
          </SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};
