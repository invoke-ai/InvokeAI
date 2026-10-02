import type { ReactNode } from 'react';

import { SegmentGroup } from '@chakra-ui/react';
import { useCallback } from 'react';

// Paint the checked item's background too; axe cannot infer the indicator sibling behind it.
const CHECKED_ITEM_STYLES = { bg: 'accent.solid', color: 'accent.contrast' } as const;

export interface SegmentedControlOption {
  disabled?: boolean;
  label: ReactNode;
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

/** The house segmented control: an `xs` group of equal centered segments with `2xs` labels. */
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
      size="xs"
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
          value={option.value}
          _checked={CHECKED_ITEM_STYLES}
        >
          <SegmentGroup.ItemHiddenInput />
          <SegmentGroup.ItemText fontSize="2xs">{option.label}</SegmentGroup.ItemText>
        </SegmentGroup.Item>
      ))}
    </SegmentGroup.Root>
  );
};
