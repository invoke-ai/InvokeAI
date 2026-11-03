import type {
  ComboboxInputProps,
  ComboboxRootProps,
  ComboboxValueChangeDetails,
  CollectionItem,
} from '@chakra-ui/react';

import { Combobox as ChakraCombobox, createListCollection, Portal, useFieldContext } from '@chakra-ui/react';
import { inputShellInteraction } from '@theme/recipes';
import { CheckIcon, ChevronDownIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, type UIEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { FieldInsideLabel, INSIDE_LABEL_INSET, useInsideFieldLabel } from './Field';
import { Scrollable } from './Scrollable';

const COMBOBOX_POSITIONING = { placement: 'bottom-start', sameWidth: true } as const;

// Zag scrolls its content element by default; the Scrollable viewport is the scroll container here.
const scrollHighlightedIntoView = ({ getElement }: { getElement: () => HTMLElement | null }) =>
  getElement()?.scrollIntoView({ block: 'nearest' });

/**
 * Under an inside label the control is the box, as InputShell is: it carries the input's border, focus, open, invalid
 * and disabled states around both the label and the input.
 */
const INSIDE_CONTROL_CSS = {
  ...inputShellInteraction,
  alignItems: 'center',
  borderColor: 'border',
  borderRadius: 'l2',
  borderWidth: '1px',
  display: 'flex',
  // The label carries the inset and the gap before the value as its own padding, full height, so a click anywhere left
  // of the input lands on the label and opens the input like the rest of the box.
  '& > label': {
    alignSelf: 'stretch',
    lineHeight: 'calc(var(--combobox-input-height) - 2px)',
    pe: '2',
    ps: INSIDE_LABEL_INSET,
  },
  _open: { borderColor: 'accent.solid' },
  // The box dims once; its input and chevron would otherwise dim again inside it.
  _disabled: { layerStyle: 'disabled', '& [data-part=input], & [data-part=indicator-group]': { opacity: 1 } },
} as const;

/** The selected label sits at the end of the box, after the inside label; typing filters as usual. */
const INSIDE_INPUT_CSS = {
  bg: 'transparent',
  borderWidth: '0',
  flex: '1',
  minW: '0',
  minH: 'calc(var(--combobox-input-height) - 2px)',
  ps: '0',
  textAlign: 'end',
  textOverflow: 'ellipsis',
} as const;

export interface ComboboxOption extends CollectionItem {
  disabled?: boolean;
  label: string;
  /** Additional fields already searched by the server, such as tags and descriptions. */
  searchText?: string;
  value: string;
}

export interface ComboboxProps extends Omit<
  ComboboxRootProps<ComboboxOption>,
  | 'allowCustomValue'
  | 'collection'
  | 'inputValue'
  | 'multiple'
  | 'onInputValueChange'
  | 'onOpenChange'
  | 'onValueChange'
  | 'open'
  | 'value'
> {
  inputProps?: ComboboxInputProps;
  noResultsText?: string;
  options: readonly ComboboxOption[];
  searchPlaceholder?: string;
  value: string | null;
  onInputValueChange?: (value: string) => void;
  onItemReselect?: (value: string) => void;
  onListScrollToBottom?: () => void;
  onValueChange: (value: string) => void;
}

/** Controlled, single-value searchable picker with workbench-standard portal positioning. */
export const Combobox = ({
  'aria-label': ariaLabel,
  inputProps,
  noResultsText,
  onValueChange,
  onItemReselect,
  options,
  onInputValueChange,
  onListScrollToBottom,
  searchPlaceholder,
  value,
  ...rootProps
}: ComboboxProps) => {
  const { t } = useTranslation();
  const field = useFieldContext();
  const insideLabel = useInsideFieldLabel();
  // Ark's input takes only the helper text from Field; an invalid field's error describes it too.
  const fieldInputProps = field?.getInputProps();
  const describedBy =
    [fieldInputProps?.['aria-describedby'], fieldInputProps?.['aria-errormessage'], inputProps?.['aria-describedby']]
      .filter(Boolean)
      .join(' ') || undefined;
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const lastNotifiedInputValue = useRef('');
  const selectedLabel = options.find((option) => option.value === value)?.label ?? '';
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const selectedValues = useMemo(() => (value ? [value] : []), [value]);
  const filteredOptions = useMemo(
    () =>
      normalizedQuery === ''
        ? options
        : options.filter(
            (option) =>
              option.label.toLocaleLowerCase().includes(normalizedQuery) ||
              option.value.toLocaleLowerCase().includes(normalizedQuery) ||
              option.searchText?.toLocaleLowerCase().includes(normalizedQuery)
          ),
    [normalizedQuery, options]
  );
  const collection = useMemo(
    () =>
      createListCollection({
        items: filteredOptions,
        isItemDisabled: (item) => item.disabled === true,
        itemToString: (item) => item.label,
        itemToValue: (item) => item.value,
      }),
    [filteredOptions]
  );
  const notifyInputValueChange = useCallback(
    (nextValue: string) => {
      if (nextValue === lastNotifiedInputValue.current) {
        return;
      }

      lastNotifiedInputValue.current = nextValue;
      onInputValueChange?.(nextValue);
    },
    [onInputValueChange]
  );
  const handleOpenChange = useCallback(
    (details: { open: boolean }) => {
      setIsOpen(details.open);
      setQuery('');

      if (!details.open) {
        notifyInputValueChange('');
      }
    },
    [notifyInputValueChange]
  );
  const handleInputValueChange = useCallback(
    (details: { inputValue: string; reason?: string }) => {
      if (details.reason === 'input-change' || details.reason === 'clear-trigger') {
        setQuery(details.inputValue);
        notifyInputValueChange(details.inputValue);
      }
    },
    [notifyInputValueChange]
  );
  const handleListScroll = useCallback(
    (event: UIEvent<HTMLDivElement>) => {
      const list = event.currentTarget;
      // The scroll area dispatches no-op scrolls while the popup is still unmeasured; only real overflow counts.
      const overflows = list.scrollHeight > list.clientHeight;

      if (overflows && list.scrollHeight - list.scrollTop - list.clientHeight <= 24) {
        onListScrollToBottom?.();
      }
    },
    [onListScrollToBottom]
  );
  const listViewportProps = useMemo(
    () => (onListScrollToBottom ? { onScroll: handleListScroll } : undefined),
    [handleListScroll, onListScrollToBottom]
  );
  const handleValueChange = useCallback(
    (details: ComboboxValueChangeDetails<ComboboxOption>) => {
      const nextValue = details.value[0];

      if (nextValue !== undefined) {
        onValueChange(nextValue);
      }
    },
    [onValueChange]
  );
  const handleSelect = useCallback(
    ({ itemValue }: { itemValue: string }) => {
      if (itemValue === value) {
        onItemReselect?.(itemValue);
      }
    },
    [onItemReselect, value]
  );

  return (
    // Mount items only while open to avoid per-combobox scroll observers at rest.
    <ChakraCombobox.Root
      lazyMount
      unmountOnExit
      {...rootProps}
      allowCustomValue={false}
      closeOnSelect
      collection={collection}
      inputBehavior="autohighlight"
      inputValue={isOpen ? query : selectedLabel}
      multiple={false}
      open={isOpen}
      openOnClick
      positioning={COMBOBOX_POSITIONING}
      scrollToIndexFn={scrollHighlightedIntoView}
      selectionBehavior="replace"
      value={selectedValues}
      onInputValueChange={handleInputValueChange}
      onOpenChange={handleOpenChange}
      onSelect={onItemReselect ? handleSelect : undefined}
      onValueChange={handleValueChange}
    >
      <ChakraCombobox.Control css={insideLabel ? INSIDE_CONTROL_CSS : undefined}>
        {insideLabel && field ? <FieldInsideLabel htmlFor={field.ids.control} label={insideLabel} /> : null}
        <ChakraCombobox.Input
          css={insideLabel ? INSIDE_INPUT_CSS : undefined}
          {...inputProps}
          aria-describedby={describedBy}
          aria-errormessage={inputProps?.['aria-errormessage'] ?? fieldInputProps?.['aria-errormessage']}
          aria-label={inputProps?.['aria-label'] ?? ariaLabel}
          placeholder={searchPlaceholder ?? t('common.searchOptions')}
        />
        <ChakraCombobox.IndicatorGroup>
          <ChakraCombobox.Trigger aria-label={t('common.openSelector')}>
            <ChevronDownIcon />
          </ChakraCombobox.Trigger>
        </ChakraCombobox.IndicatorGroup>
      </ChakraCombobox.Control>
      <Portal>
        <ChakraCombobox.Positioner>
          <ChakraCombobox.Content>
            <Scrollable maxH="16rem" viewportProps={listViewportProps}>
              <ChakraCombobox.List>
                {collection.items.map((item) => (
                  <ChakraCombobox.Item key={item.value} item={item}>
                    <ChakraCombobox.ItemText>{item.label}</ChakraCombobox.ItemText>
                    <ChakraCombobox.ItemIndicator>
                      <CheckIcon />
                    </ChakraCombobox.ItemIndicator>
                  </ChakraCombobox.Item>
                ))}
                <ChakraCombobox.Empty color="fg.muted" fontSize="md" px="3" py="2">
                  {noResultsText ?? t('common.noOptionsFound')}
                </ChakraCombobox.Empty>
              </ChakraCombobox.List>
            </Scrollable>
          </ChakraCombobox.Content>
        </ChakraCombobox.Positioner>
      </Portal>
    </ChakraCombobox.Root>
  );
};
