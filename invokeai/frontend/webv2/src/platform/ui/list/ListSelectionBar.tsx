import type { ReactNode } from 'react';

import { Checkbox, HStack, Separator, Text, type StackProps } from '@chakra-ui/react';
import { Children } from 'react';

import { LIST_CHECK_INSET, LIST_ROW_INSET } from './listLayout';

export interface ListSelectionBarProps extends Omit<StackProps, 'children'> {
  /** `'indeterminate'` when some, but not every, matching row is selected. */
  checked: boolean | 'indeterminate';
  isDisabled?: boolean;
  /** Visible and accessible name of the select-all control. */
  label: string;
  onCheckedChange: () => void;
  /** Selection count or estimate, right-aligned; muted so the actions read first. */
  summary?: string;
  /** Lets an action describe itself with the summary via aria-describedby. */
  summaryId?: string;
  /** Actions for the selection, after a separator; omit while nothing is selected. */
  children?: ReactNode;
}

/**
 * Select-all row above a multi-select list. Keep it mounted whatever the selection so rows never shift beneath
 * it; toggle the actions instead. Its insets put the checkbox in the rows' checkbox column.
 */
export const ListSelectionBar = ({
  checked,
  children,
  isDisabled = false,
  label,
  summary,
  summaryId,
  onCheckedChange,
  ...stackProps
}: ListSelectionBarProps) => (
  <HStack
    borderBottomWidth="1px"
    borderColor="border.subtle"
    flexShrink={0}
    gap="2"
    minH="8"
    pe={LIST_ROW_INSET}
    ps={LIST_CHECK_INSET}
    py="1.5"
    {...stackProps}
  >
    <Checkbox.Root
      aria-label={label}
      checked={checked}
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
        {children}
      </>
    ) : null}
  </HStack>
);
