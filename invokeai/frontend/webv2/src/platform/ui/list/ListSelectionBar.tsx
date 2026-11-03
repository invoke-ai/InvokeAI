import type { ReactNode } from 'react';

import { Checkbox, HStack, Separator, Text, type StackProps } from '@chakra-ui/react';
import { Children, useId, useMemo } from 'react';

import {
  LIST_ACTIONS_PADDING,
  LIST_CHECK_INSET,
  LIST_CHECK_PADDING,
  LIST_ROW_INSET,
  type ListInset,
} from './listLayout';

export interface ListSelectionBarProps extends Omit<StackProps, 'children'> {
  /** `'indeterminate'` when some, but not every, matching row is selected. */
  checked: boolean | 'indeterminate';
  isDisabled?: boolean;
  /** Visible and accessible name of the select-all control. */
  label: string;
  /** Match the `inset` of the List below so the checkbox and actions stay in the rows' columns. */
  inset?: ListInset;
  onCheckedChange: () => void;
  /** Selection count or estimate, right-aligned; muted so the actions read first. */
  summary?: string;
  /** Lets an action describe itself with the summary via aria-describedby. */
  summaryId?: string;
  /** Actions for the selection, after a separator; omit while nothing is selected. Size `sm` or smaller. */
  children?: ReactNode;
}

/** An `sm` control, `py="1"` either side of it, and the bottom border. */
const BAR_HEIGHT = 'calc({sizes.control.sm} + {spacing.2} + 1px)';

/**
 * Select-all row above a multi-select list. Keep it mounted whatever the selection so rows never shift beneath
 * it; toggle the actions instead. Its height is fixed to fit actions of size `sm` or smaller, so it is as tall with
 * actions as without them; keep every action at `sm` or below. Its insets put the checkbox in the rows' checkbox column.
 */
export const ListSelectionBar = ({
  checked,
  children,
  inset = 'row',
  isDisabled = false,
  label,
  summary,
  summaryId,
  onCheckedChange,
  ...stackProps
}: ListSelectionBarProps) => {
  // Its own ids, so a list inside a form Field never binds the select-all control to the Field's input.
  const inputId = useId();
  const checkboxIds = useMemo(() => ({ hiddenInput: inputId }), [inputId]);

  return (
    <HStack
      borderBottomWidth="1px"
      borderColor="border.subtle"
      flexShrink={0}
      gap="2"
      h={BAR_HEIGHT}
      pe={inset === 'none' ? LIST_ACTIONS_PADDING : `calc(${LIST_ROW_INSET} + ${LIST_ACTIONS_PADDING})`}
      ps={inset === 'none' ? LIST_CHECK_PADDING : LIST_CHECK_INSET}
      py="1"
      {...stackProps}
    >
      <Checkbox.Root
        aria-label={label}
        checked={checked}
        ids={checkboxIds}
        colorPalette="accent"
        disabled={isDisabled}
        size="sm"
        onCheckedChange={onCheckedChange}
      >
        <Checkbox.HiddenInput />
        <Checkbox.Control />
        <Checkbox.Label color="fg.muted" fontSize="xs" fontWeight="600">
          {label}
        </Checkbox.Label>
      </Checkbox.Root>
      <Text color="fg.muted" flex="1" fontSize="xs" id={summaryId} minW="0" textAlign="end" truncate>
        {summary}
      </Text>
      {Children.toArray(children).length > 0 ? (
        <>
          <Separator borderColor="border.subtle" h="4" orientation="vertical" />
          {/* Adjacent actions sit as close as a row's actions do. */}
          <HStack gap="0.5">{children}</HStack>
        </>
      ) : null}
    </HStack>
  );
};
