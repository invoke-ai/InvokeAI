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

import { Portal, Select as ChakraSelect, useFieldContext } from '@chakra-ui/react';
import { useCallback, useRef } from 'react';

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
  /** The visible header for a group key; defaults to the key itself. */
  renderGroupLabel?: (group: string) => ReactNode;
  getItemKey?: (item: T, index: number) => Key;
  indicatorGroupProps?: SelectIndicatorGroupProps;
  itemIndicator?: boolean;
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
      scrollToIndexFn={itemsMaxH ? scrollToIndexFn : undefined}
      unmountOnExit
      {...rootProps}
      // An empty list is nothing to choose from: the trigger disables instead of opening a blank menu.
      disabled={rootProps.disabled || collection.items.length === 0}
    >
      {/* Materialize the trigger's aria-labelledby target; aria-label on Root would name a generic div instead. */}
      {ariaLabel && !field ? <ChakraSelect.Label srOnly>{ariaLabel}</ChakraSelect.Label> : null}
      <ChakraSelect.HiddenSelect />
      <ChakraSelect.Control>
        <ChakraSelect.Trigger {...triggerProps}>
          <ChakraSelect.ValueText {...valueTextProps}>{valueText}</ChakraSelect.ValueText>
        </ChakraSelect.Trigger>
        <ChakraSelect.IndicatorGroup {...indicatorGroupProps}>
          <ChakraSelect.Indicator />
        </ChakraSelect.IndicatorGroup>
      </ChakraSelect.Control>
      <Portal disabled={!portalled}>
        <ChakraSelect.Positioner {...positionerProps}>
          <ChakraSelect.Content {...contentProps}>
            <SelectItems ref={itemsRef} maxH={itemsMaxH ?? undefined}>
              {groupBy
                ? partitionByGroup(collection.items, groupBy).map(({ group, items, startIndex }) => (
                    <ChakraSelect.ItemGroup key={group}>
                      <ChakraSelect.ItemGroupLabel>
                        {renderGroupLabel ? renderGroupLabel(group) : group}
                      </ChakraSelect.ItemGroupLabel>
                      {items.map((item, offset) => (
                        <ChakraSelect.Item key={getItemKey(item, startIndex + offset)} item={item}>
                          <ChakraSelect.ItemText>{renderItem(item)}</ChakraSelect.ItemText>
                          {itemIndicator ? <ChakraSelect.ItemIndicator /> : null}
                        </ChakraSelect.Item>
                      ))}
                    </ChakraSelect.ItemGroup>
                  ))
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
