import type {
  CollectionItem,
  SelectContentProps,
  SelectIndicatorGroupProps,
  SelectPositionerProps,
  SelectRootProps,
  SelectTriggerProps,
  SelectValueTextProps,
} from '@chakra-ui/react';
import type { Key, ReactNode } from 'react';

import { Portal, Select as ChakraSelect, Separator, useFieldContext } from '@chakra-ui/react';
import { Fragment, useCallback, useRef } from 'react';

import { FieldInsideLabel, INSIDE_LABEL_INSET, useInsideFieldLabel } from './Field';
import { Scrollable } from './Scrollable';

const getDefaultItemKey = <T extends CollectionItem>(item: T, index: number): Key => {
  const keyedItem = item as { id?: Key; value?: Key };

  return keyedItem.value ?? keyedItem.id ?? index;
};

const renderDefaultItem = <T extends CollectionItem>(item: T): ReactNode => {
  const labelledItem = item as { label?: ReactNode; value?: ReactNode };

  return labelledItem.label ?? labelledItem.value;
};

/** Consecutive same-group runs, keeping the collection's flat order and indices. */
const partitionByGroup = <T,>(
  items: readonly T[],
  groupBy: (item: T) => string
): { group: string; items: T[]; startIndex: number }[] => {
  const runs: { group: string; items: T[]; startIndex: number }[] = [];
  items.forEach((item, index) => {
    const group = groupBy(item);
    const last = runs[runs.length - 1];
    if (last && last.group === group) {
      last.items.push(item);
    } else {
      runs.push({ group, items: [item], startIndex: index });
    }
  });
  return runs;
};

/**
 * With the chevron in the trigger's flow (an inside label, or a trigger sized to its value) the value fills the room
 * left before it, end-aligned and truncating on one line; rich value text (an icon and a name) aligns as one inline box
 * that can still shrink.
 */
const IN_FLOW_VALUE_TEXT_CSS = {
  display: 'block',
  flex: '1',
  maxW: 'none',
  minW: '0',
  overflow: 'hidden',
  textAlign: 'end',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
  '& > *': { display: 'inline-flex', maxW: 'full', verticalAlign: 'middle' },
} as const;

/** The trigger already dims when disabled; the chevron inside it keeps its resting colour rather than dimming twice. */
const IN_TRIGGER_INDICATOR_CSS = { _disabled: { color: 'fg.muted' } } as const;

/** Fits the anchor's available space with the same headroom the settings selects use. */
const DEFAULT_ITEMS_MAX_H = 'min(20rem, calc(var(--available-height) - 0.5rem))';

export interface SelectProps<T extends CollectionItem> extends Omit<SelectRootProps<T>, 'children'> {
  contentProps?: SelectContentProps;
  /**
   * Caps menu height and routes keyboard scrolling through Scrollable; null restores the machine's native
   * overflow.
   */
  itemsMaxH?: string | null;
  /** Only consecutive equal keys form a group; callers must order items by group. */
  groupBy?: (item: T) => string;
  /**
   * The visible header for a group key; defaults to the key itself. `null` renders the group without a header,
   * under a divider unless it opens the list.
   */
  renderGroupLabel?: (group: string) => ReactNode;
  getItemKey?: (item: T, index: number) => Key;
  indicatorGroupProps?: SelectIndicatorGroupProps;
  itemIndicator?: boolean;
  /**
   * Sizes the select to its value instead of its container's width, for a select trailing its label in a row; cap it
   * with `maxW`. The chevron joins the trigger's flow, so a truncated value stops before it.
   */
  fitContent?: boolean;
  portalled?: boolean;
  positionerProps?: SelectPositionerProps;
  renderItem?: (item: T) => ReactNode;
  triggerProps?: SelectTriggerProps;
  valueText?: ReactNode;
  valueTextProps?: SelectValueTextProps;
}

/** Workbench Select: Chakra's custom Select with the standard trigger, portal, and item markup pre-wired. */
export const Select = <T extends CollectionItem>({
  'aria-label': ariaLabel,
  collection,
  contentProps,
  getItemKey = getDefaultItemKey,
  groupBy,
  indicatorGroupProps,
  itemsMaxH = DEFAULT_ITEMS_MAX_H,
  itemIndicator = true,
  fitContent = false,
  portalled = true,
  positionerProps,
  renderGroupLabel,
  renderItem = renderDefaultItem,
  triggerProps,
  valueText,
  valueTextProps,
  ...rootProps
}: SelectProps<T>) => {
  // Inside Field, reuse its label ID; rendering another Label would duplicate it.
  const field = useFieldContext();
  const insideLabel = useInsideFieldLabel();
  // An inside label or a content-sized trigger keeps the chevron in the trigger's flow, so the value truncates
  // before it (or the trigger grows to hold it) instead of running under it.
  const isIndicatorInTrigger = insideLabel !== null || fitContent;
  // Zag names the trigger by the field's label but leaves its help and error text to the hidden select.
  const fieldControlProps = field?.getInputProps();
  const describedBy =
    [
      fieldControlProps?.['aria-describedby'],
      fieldControlProps?.['aria-errormessage'],
      triggerProps?.['aria-describedby'],
    ]
      .filter(Boolean)
      .join(' ') || undefined;
  const itemsRef = useRef<HTMLDivElement>(null);
  const scrollToIndexFn = useCallback(({ index }: { index: number }) => {
    const options = itemsRef.current?.querySelectorAll('[role="option"]');
    options?.[index]?.scrollIntoView({ block: 'nearest' });
  }, []);
  return (
    // Mount items only while open to avoid per-select scroll observers at rest.
    <ChakraSelect.Root
      collection={collection}
      lazyMount
      minW={fitContent ? '0' : undefined}
      scrollToIndexFn={itemsMaxH ? scrollToIndexFn : undefined}
      unmountOnExit
      w={fitContent ? 'auto' : undefined}
      {...rootProps}
      // An empty list is nothing to choose from: the trigger disables instead of opening a blank menu.
      // An explicit prop would otherwise override the enclosing Field's disabled state.
      disabled={rootProps.disabled || field?.disabled || collection.items.length === 0}
    >
      {/* Materialize the trigger's aria-labelledby target; aria-label on Root would name a generic div instead. */}
      {ariaLabel && !field ? <ChakraSelect.Label srOnly>{ariaLabel}</ChakraSelect.Label> : null}
      <ChakraSelect.HiddenSelect />
      <ChakraSelect.Control>
        <ChakraSelect.Trigger
          ps={insideLabel ? INSIDE_LABEL_INSET : undefined}
          {...triggerProps}
          aria-describedby={describedBy}
        >
          {insideLabel ? <FieldInsideLabel label={insideLabel} /> : null}
          <ChakraSelect.ValueText css={isIndicatorInTrigger ? IN_FLOW_VALUE_TEXT_CSS : undefined} {...valueTextProps}>
            {valueText}
          </ChakraSelect.ValueText>
          {isIndicatorInTrigger ? <ChakraSelect.Indicator css={IN_TRIGGER_INDICATOR_CSS} /> : null}
        </ChakraSelect.Trigger>
        {isIndicatorInTrigger ? null : (
          <ChakraSelect.IndicatorGroup {...indicatorGroupProps}>
            <ChakraSelect.Indicator />
          </ChakraSelect.IndicatorGroup>
        )}
      </ChakraSelect.Control>
      <Portal disabled={!portalled}>
        <ChakraSelect.Positioner {...positionerProps}>
          <ChakraSelect.Content {...contentProps}>
            <SelectItems ref={itemsRef} maxH={itemsMaxH ?? undefined}>
              {groupBy
                ? partitionByGroup(collection.items, groupBy).map(({ group, items, startIndex }) => {
                    const label = renderGroupLabel ? renderGroupLabel(group) : group;
                    const groupItems = items.map((item, offset) => (
                      <ChakraSelect.Item key={getItemKey(item, startIndex + offset)} item={item}>
                        <ChakraSelect.ItemText>{renderItem(item)}</ChakraSelect.ItemText>
                        {itemIndicator ? <ChakraSelect.ItemIndicator /> : null}
                      </ChakraSelect.Item>
                    ));
                    // A header-less group is not a labelled listbox group: its options follow a divider, which
                    // stays out of the accessibility tree because a listbox owns only options and groups.
                    return label === null ? (
                      <Fragment key={group}>
                        {startIndex > 0 ? <Separator aria-hidden borderColor="border.subtle" mx="-1" my="1" /> : null}
                        {groupItems}
                      </Fragment>
                    ) : (
                      <ChakraSelect.ItemGroup key={group}>
                        <ChakraSelect.ItemGroupLabel>{label}</ChakraSelect.ItemGroupLabel>
                        {groupItems}
                      </ChakraSelect.ItemGroup>
                    );
                  })
                : collection.items.map((item, index) => (
                    <ChakraSelect.Item key={getItemKey(item, index)} item={item}>
                      <ChakraSelect.ItemText>{renderItem(item)}</ChakraSelect.ItemText>
                      {itemIndicator ? <ChakraSelect.ItemIndicator /> : null}
                    </ChakraSelect.Item>
                  ))}
            </SelectItems>
          </ChakraSelect.Content>
        </ChakraSelect.Positioner>
      </Portal>
    </ChakraSelect.Root>
  );
};

/** Items pass through untouched unless a max height asks for a scroll viewport. */
const SelectItems = ({
  children,
  maxH,
  ref,
}: {
  children: ReactNode;
  maxH?: string;
  ref: React.RefObject<HTMLDivElement | null>;
}) =>
  maxH ? (
    <Scrollable ref={ref} maxH={maxH}>
      {children}
    </Scrollable>
  ) : (
    children
  );
