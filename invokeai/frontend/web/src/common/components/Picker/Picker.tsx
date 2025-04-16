import type { InputProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Divider, Flex, Input, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import { fixedForwardRef } from 'common/util/fixedForwardRef';
import { typedMemo } from 'common/util/typedMemo';
import type { ChangeEvent } from 'react';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type Group<T extends object, U = any> = {
  id: string;
  data: U;
  options: T[];
};

const isGroup = <T extends object>(option: T | Group<T>): option is Group<T> => {
  return option ? 'options' in option && Array.isArray(option.options) : false;
};

export type ImperativeModelPickerHandle = {
  inputRef: React.RefObject<HTMLInputElement>;
  rootRef: React.RefObject<HTMLDivElement>;
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
};

const DefaultOptionComponent = typedMemo(<T extends object>({ option }: { option: T }) => {
  const { getOptionId } = usePickerContext();
  return <Text fontWeight="bold">{getOptionId(option)}</Text>;
});
DefaultOptionComponent.displayName = 'DefaultOptionComponent';

const DefaultGroupHeaderComponent = typedMemo(<T extends object>({ group }: { group: Group<T> }) => {
  return <Text fontWeight="bold">{group.id}</Text>;
});
DefaultGroupHeaderComponent.displayName = 'DefaultGroupHeaderComponent';

const DefaultNoOptionsFallbackComponent = typedMemo(() => {
  const { t } = useTranslation();
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      <Text variant="subtext">{t('common.noOptions')}</Text>
    </Flex>
  );
});
DefaultNoOptionsFallbackComponent.displayName = 'DefaultNoOptionsFallbackComponent';

const DefaultNoMatchesFallbackComponent = typedMemo(() => {
  const { t } = useTranslation();
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      <Text variant="subtext">{t('common.noMatches')}</Text>
    </Flex>
  );
});
DefaultNoMatchesFallbackComponent.displayName = 'DefaultNoMatchesFallbackComponent';

export type PickerProps<T extends object> = {
  options: (T | Group<T>)[];
  getOptionId: (option: T) => string;
  isMatch: (option: T, searchTerm: string) => boolean;
  getIsDisabled?: (option: T) => boolean;
  selectedItem?: T;
  onSelect?: (option: T) => void;
  onClose?: () => void;
  handleRef?: React.Ref<ImperativeModelPickerHandle>;
  SearchBarComponent?: ReturnType<typeof fixedForwardRef<HTMLInputElement, InputProps>>;
  NoOptionsFallbackComponent?: React.ComponentType;
  NoMatchesFallbackComponent?: React.ComponentType;
  OptionComponent?: React.ComponentType<{ option: T }>;
  GroupHeaderComponent?: React.ComponentType<{ group: Group<T> }>;
};

type PickerContextState<T extends object> = {
  getOptionId: (option: T) => string;
  isMatch: (option: T, searchTerm: string) => boolean;
  getIsDisabled?: (option: T) => boolean;
  setActiveOptionId: (id: string) => void;
  onSelectById: (id: string) => void;
  SearchBarComponent: ReturnType<typeof fixedForwardRef<HTMLInputElement, InputProps>>;
  NoOptionsFallbackComponent: React.ComponentType;
  NoMatchesFallbackComponent: React.ComponentType;
  OptionComponent: React.ComponentType<{ option: T }>;
  GroupHeaderComponent: React.ComponentType<{ group: Group<T> }>;
};

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const PickerContext = createContext<PickerContextState<any> | null>(null);
const usePickerContext = <T extends object>(): PickerContextState<T> => {
  const context = useContext(PickerContext);
  assert(context !== null, 'usePickerContext must be used within a PickerProvider');
  return context;
};

export const getRegex = (searchTerm: string) => {
  const terms = searchTerm
    .trim()
    .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
    .split(' ')
    .filter((term) => term.length > 0);

  if (terms.length === 0) {
    return new RegExp('', 'gi');
  }

  // Create positive lookaheads for each term - matches in any order
  const pattern = terms.map((term) => `(?=.*${term})`).join('');
  return new RegExp(`${pattern}.+`, 'i');
};

const getFirstOption = <T extends object>(options: (T | Group<T>)[]): T | undefined => {
  const firstOptionOrGroup = options[0];
  if (!firstOptionOrGroup) {
    return;
  }
  if (isGroup(firstOptionOrGroup)) {
    return firstOptionOrGroup.options[0];
  } else {
    return firstOptionOrGroup;
  }
};

const getFirstOptionId = <T extends object>(
  options: (T | Group<T>)[],
  getOptionId: (item: T) => string
): string | undefined => {
  const firstOptionOrGroup = getFirstOption(options);
  if (firstOptionOrGroup) {
    return getOptionId(firstOptionOrGroup);
  } else {
    return undefined;
  }
};

const findOption = <T extends object>(
  options: (T | Group<T>)[],
  id: string,
  getOptionId: (item: T) => string
): T | undefined => {
  for (const optionOrGroup of options) {
    if (isGroup(optionOrGroup)) {
      const option = optionOrGroup.options.find((opt) => getOptionId(opt) === id);
      if (option) {
        return option;
      }
    } else {
      if (getOptionId(optionOrGroup) === id) {
        return optionOrGroup;
      }
    }
  }
};

const flattenOptions = <T extends object>(options: (T | Group<T>)[]): T[] => {
  const flattened: T[] = [];
  for (const optionOrGroup of options) {
    if (isGroup(optionOrGroup)) {
      flattened.push(...optionOrGroup.options);
    } else {
      flattened.push(optionOrGroup);
    }
  }
  return flattened;
};

export const Picker = typedMemo(<T extends object>(props: PickerProps<T>) => {
  const {
    getOptionId,
    options,
    handleRef,
    isMatch,
    getIsDisabled,
    onClose,
    onSelect,
    selectedItem,
    SearchBarComponent = DefaultPickerSearchBarComponent,
    NoMatchesFallbackComponent = DefaultNoMatchesFallbackComponent,
    NoOptionsFallbackComponent = DefaultNoOptionsFallbackComponent,
    OptionComponent = DefaultOptionComponent,
    GroupHeaderComponent = DefaultGroupHeaderComponent,
  } = props;
  const [activeOptionId, setActiveOptionId, getActiveOptionId] = useStateImperative(() =>
    getFirstOptionId(options, getOptionId)
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [filteredOptions, setFilteredOptions] = useState<(T | Group<T>)[]>(options);
  const flattenedOptions = useMemo(() => flattenOptions(options), [options]);
  const flattenedFilteredOptions = useMemo(() => flattenOptions(filteredOptions), [filteredOptions]);
  const [searchTerm, setSearchTerm] = useState('');
  useImperativeHandle(handleRef, () => ({ inputRef, rootRef, searchTerm, setSearchTerm }), [searchTerm]);

  const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  useEffect(() => {
    if (!searchTerm) {
      setFilteredOptions(options);
      setActiveOptionId(getFirstOptionId(options, getOptionId));
    } else {
      const lowercasedSearchTerm = searchTerm.toLowerCase();
      const filtered: (T | Group<T>)[] = [];
      for (const item of props.options) {
        if (isGroup(item)) {
          const filteredItems = item.options.filter((item) => isMatch(item, lowercasedSearchTerm));
          if (filteredItems.length > 0) {
            filtered.push({ ...item, options: filteredItems });
          }
        } else {
          if (isMatch(item, searchTerm)) {
            filtered.push(item);
          }
        }
      }
      setFilteredOptions(filtered);
      setActiveOptionId(getFirstOptionId(filtered, getOptionId));
    }
  }, [searchTerm, setActiveOptionId, props.options, options, getOptionId, isMatch]);

  const onSelectById = useCallback(
    (id: string) => {
      const item = findOption(options, id, getOptionId);
      if (!item) {
        // Model not found? We should never get here.
        return;
      }
      onSelect?.(item);
    },
    [getOptionId, options, onSelect]
  );

  const setValueAndScrollIntoView = useCallback(
    (id: string) => {
      setActiveOptionId(id);
      const rootEl = rootRef.current;
      if (!rootEl) {
        return;
      }
      const itemEl = rootEl.querySelector(`#${CSS.escape(id)}`);
      if (!itemEl) {
        return;
      }
      itemEl.scrollIntoView({ block: 'nearest' });
    },
    [setActiveOptionId]
  );

  const prev = useCallback(
    (e: React.KeyboardEvent) => {
      e.preventDefault();
      const activeOptionId = getActiveOptionId();
      if (flattenedFilteredOptions.length === 0) {
        return;
      }
      if (e.metaKey) {
        const item = flattenedFilteredOptions.at(0);
        if (item) {
          setValueAndScrollIntoView(getOptionId(item));
        }
        return;
      }
      const currentIndex = flattenedFilteredOptions.findIndex((item) => getOptionId(item) === activeOptionId);
      if (currentIndex < 0) {
        return;
      }
      let newIndex = currentIndex - 1;
      if (newIndex < 0) {
        newIndex = flattenedFilteredOptions.length - 1;
      }
      const item = flattenedFilteredOptions.at(newIndex);
      if (item) {
        setValueAndScrollIntoView(getOptionId(item));
      }
    },
    [getActiveOptionId, flattenedFilteredOptions, setValueAndScrollIntoView, getOptionId]
  );

  const next = useCallback(
    (e: React.KeyboardEvent) => {
      e.preventDefault();
      const activeOptionId = getActiveOptionId();
      if (flattenedFilteredOptions.length === 0) {
        return;
      }
      if (e.metaKey) {
        const item = flattenedFilteredOptions.at(-1);
        if (item) {
          setValueAndScrollIntoView(getOptionId(item));
        }
        return;
      }

      const currentIndex = flattenedFilteredOptions.findIndex((item) => getOptionId(item) === activeOptionId);
      if (currentIndex < 0) {
        return;
      }
      let newIndex = currentIndex + 1;
      if (newIndex >= flattenedFilteredOptions.length) {
        newIndex = 0;
      }
      const item = flattenedFilteredOptions.at(newIndex);
      if (item) {
        setValueAndScrollIntoView(getOptionId(item));
      }
    },
    [getActiveOptionId, flattenedFilteredOptions, setValueAndScrollIntoView, getOptionId]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowUp') {
        prev(e);
      } else if (e.key === 'ArrowDown') {
        next(e);
      } else if (e.key === 'Enter') {
        const activeOptionId = getActiveOptionId();
        if (!activeOptionId) {
          return;
        }
        onSelectById(activeOptionId);
      } else if (e.key === 'Escape') {
        onClose?.();
      } else if (e.key === '/') {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    },
    [prev, next, getActiveOptionId, onSelectById, onClose]
  );

  const ctx = useMemo(
    () =>
      ({
        getOptionId,
        isMatch,
        getIsDisabled,
        onSelectById,
        setActiveOptionId,
        SearchBarComponent,
        NoOptionsFallbackComponent,
        NoMatchesFallbackComponent,
        OptionComponent,
        GroupHeaderComponent,
      }) satisfies PickerContextState<T>,
    [
      getOptionId,
      isMatch,
      getIsDisabled,
      onSelectById,
      setActiveOptionId,
      SearchBarComponent,
      NoOptionsFallbackComponent,
      NoMatchesFallbackComponent,
      OptionComponent,
      GroupHeaderComponent,
    ]
  );

  return (
    <PickerContext.Provider value={ctx}>
      <Flex
        tabIndex={-1}
        ref={rootRef}
        flexGrow={1}
        flexDir="column"
        p={2}
        w="full"
        h="full"
        gap={2}
        onKeyDown={onKeyDown}
      >
        <SearchBarComponent ref={inputRef} value={searchTerm} onChange={onChangeSearchTerm} />
        <Divider />
        <Flex tabIndex={-1} w="full" flexGrow={1}>
          {flattenedOptions.length === 0 && <NoOptionsFallbackComponent />}
          {flattenedOptions.length > 0 && flattenedFilteredOptions.length === 0 && <NoMatchesFallbackComponent />}
          {flattenedOptions.length > 0 && flattenedFilteredOptions.length > 0 && (
            <ScrollableContent>
              <PickerList
                items={filteredOptions}
                activeOptionId={activeOptionId}
                selectedItemId={selectedItem ? getOptionId(selectedItem) : undefined}
              />
            </ScrollableContent>
          )}
        </Flex>
      </Flex>
    </PickerContext.Provider>
  );
});
Picker.displayName = 'Picker';

const DefaultPickerSearchBarComponent = typedMemo(
  fixedForwardRef<HTMLInputElement, InputProps>((props, ref) => {
    return <Input placeholder="Search" ref={ref} {...props} />;
  })
);
DefaultPickerSearchBarComponent.displayName = 'DefaultPickerSearchBarComponent';

const PickerList = typedMemo(
  <T extends object>({
    items,
    activeOptionId,
    selectedItemId,
  }: {
    items: (T | Group<T>)[];
    activeOptionId: string | undefined;
    selectedItemId: string | undefined;
  }) => {
    const { getOptionId, getIsDisabled } = usePickerContext<T>();

    if (items.length === 0) {
      return (
        <IAINoContentFallback
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          icon={null}
          label="No matching models"
        />
      );
    }
    return (
      <Flex flexDir="column" w="full" gap={1}>
        {items.map((itemOrGroup) => {
          if (isGroup(itemOrGroup)) {
            return (
              <PickerOptionGroup
                key={itemOrGroup.id}
                group={itemOrGroup}
                activeOptionId={activeOptionId}
                selectedItemId={selectedItemId}
              />
            );
          } else {
            const id = getOptionId(itemOrGroup);
            return (
              <PickerOption
                key={id}
                id={id}
                option={itemOrGroup}
                isActive={id === activeOptionId}
                isSelected={id === selectedItemId}
                isDisabled={getIsDisabled?.(itemOrGroup) ?? false}
              />
            );
          }
        })}
      </Flex>
    );
  }
);
PickerList.displayName = 'PickerList';

const PickerOptionGroup = typedMemo(
  <T extends object>({
    group,
    activeOptionId,
    selectedItemId,
  }: {
    group: Group<T>;
    activeOptionId: string | undefined;
    selectedItemId: string | undefined;
  }) => {
    const { getOptionId, GroupHeaderComponent, getIsDisabled } = usePickerContext<T>();

    return (
      <Flex flexDir="column" gap={2} w="full">
        <GroupHeaderComponent group={group} />
        <Flex flexDir="column" gap={1} w="full">
          {group.options.map((item) => {
            const id = getOptionId(item);
            return (
              <PickerOption
                key={id}
                id={id}
                option={item}
                isActive={id === activeOptionId}
                isSelected={id === selectedItemId}
                isDisabled={getIsDisabled?.(item) ?? false}
              />
            );
          })}
        </Flex>
      </Flex>
    );
  }
);
PickerOptionGroup.displayName = 'PickerOptionGroup';

const itemSx: SystemStyleObject = {
  display: 'flex',
  flexDir: 'column',
  p: 2,
  cursor: 'pointer',
  borderRadius: 'base',
  '&[data-selected="true"]': {
    borderColor: 'invokeBlue.300',
    borderWidth: 1,
  },
  '&[data-active="true"]': {
    bg: 'base.700',
  },
  '&[data-disabled="true"]': {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
};

const PickerOption = typedMemo(
  <T extends object>(props: { id: string; option: T; isActive: boolean; isSelected: boolean; isDisabled: boolean }) => {
    const { OptionComponent, setActiveOptionId, onSelectById } = usePickerContext<T>();
    const { id, option, isActive, isDisabled, isSelected } = props;
    const onPointerMove = useCallback(() => {
      setActiveOptionId(id);
    }, [id, setActiveOptionId]);
    const onClick = useCallback(() => {
      onSelectById(id);
    }, [id, onSelectById]);
    return (
      <Box
        role="option"
        sx={itemSx}
        id={id}
        data-disabled={isDisabled}
        data-selected={isSelected}
        data-active={isActive}
        onPointerMove={isDisabled ? undefined : onPointerMove}
        onClick={isDisabled ? undefined : onClick}
      >
        <OptionComponent option={option} />
      </Box>
    );
  }
);
PickerOption.displayName = 'PickerOption';
