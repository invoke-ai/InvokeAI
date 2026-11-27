import type { ElementType, ReactNode } from 'react';

import { chakra, Icon, SegmentGroup } from '@chakra-ui/react';
import { useCallback } from 'react';

import { Tooltip } from './Tooltip';

export interface SegmentedControlOption {
  disabled?: boolean;
  /** Shown as the segment's text; with `icon`, the segment's accessible name and tooltip instead. */
  label: ReactNode;
  /** Makes the segment icon-only. */
  icon?: ElementType;
  value: string;
}

export interface SegmentedControlProps extends Omit<SegmentGroup.RootProps, 'onChange' | 'onValueChange' | 'value'> {
  ariaLabel?: string;
  disabled?: boolean;
  /** Default true: the control fills its container, split into equal segments. */
  isFullWidth?: boolean;
  onChange: (value: string) => void;
  options: readonly SegmentedControlOption[];
  value: string | null;
}

/** The house segmented control: an `md` group of equal centered segments with `xs` labels or tooltipped icons. */
export const SegmentedControl = ({
  ariaLabel,
  disabled,
  isFullWidth = true,
  onChange,
  options,
  value,
  ...rest
}: SegmentedControlProps) => {
  const handleValueChange = useCallback(
    ({ value: next }: SegmentGroup.ValueChangeDetails) => {
      if (next !== null) {
        onChange(next);
      }
    },
    [onChange]
  );

  return (
    <SegmentGroup.Root
      aria-label={ariaLabel}
      disabled={disabled}
      value={value}
      w={isFullWidth ? 'full' : undefined}
      onValueChange={handleValueChange}
      {...rest}
    >
      <SegmentGroup.Indicator />
      {options.map((option) => (
        <SegmentGroup.Item
          key={option.value}
          disabled={option.disabled}
          flex={isFullWidth ? '1' : undefined}
          justifyContent={isFullWidth ? 'center' : undefined}
          minW={isFullWidth ? '0' : undefined}
          px={option.icon ? '0' : undefined}
          value={option.value}
        >
          {option.icon ? (
            // The tip's trigger is inside the item, never the item itself: a trigger's data attributes would replace
            // the item's checked state. Focus on the radio inside it still reaches the trigger. Arrow keys check the
            // next radio with a click, which must not close its tip; a pointer press still does.
            <Tooltip closeOnClick={false} content={option.label}>
              <chakra.span alignItems="center" alignSelf="stretch" display="flex" flex="1" justifyContent="center">
                <SegmentGroup.ItemHiddenInput />
                <Icon as={option.icon} aria-hidden boxSize="3.5" />
              </chakra.span>
            </Tooltip>
          ) : (
            <SegmentGroup.ItemHiddenInput />
          )}
          <SegmentGroup.ItemText fontSize="xs" srOnly={option.icon ? true : undefined}>
            {option.label}
          </SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};
