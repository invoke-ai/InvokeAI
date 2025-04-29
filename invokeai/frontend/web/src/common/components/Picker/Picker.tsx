import type { BoxProps, InputProps } from '@invoke-ai/ui-library';
import { Flex, Input, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import { fixedForwardRef } from 'common/util/fixedForwardRef';
import { typedMemo } from 'common/util/typedMemo';
import type { ChangeEvent, PropsWithChildren } from 'react';
import type React from 'react';
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

const DefaultGroupComponent = typedMemo(
  <T extends object>({ group, children }: PropsWithChildren<{ group: Group<T> }>) => {
    return (
      <Flex flexDir="column" gap={2} w="full">
        <Text fontWeight="bold">{group.id}</Text>
        <Flex flexDir="column" gap={1} w="full">
          {children}
        </Flex>
      </Flex>
    );
  }
);
DefaultGroupComponent.displayName = 'DefaultGroupComponent';

const NoOptionsFallbackWrapper = typedMemo(({ children }: PropsWithChildren) => {
  const { t } = useTranslation();
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      {typeof children === 'string' ? (
        <Text variant="subtext">{children}</Text>
      ) : (
        (children ?? <Text variant="subtext">{t('common.noOptions')}</Text>)
      )}
    </Flex>
  );
});
NoOptionsFallbackWrapper.displayName = 'NoOptionsFallbackWrapper';

const NoMatchesFallbackWrapper = typedMemo(({ children }: PropsWithChildren) => {
  const { t } = useTranslation();
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      {typeof children === 'string' ? (
        <Text variant="subtext">{children}</Text>
      ) : (
        (children ?? <Text variant="subtext">{t('common.noMatches')}</Text>)
      )}
    </Flex>
  );
});
NoMatchesFallbackWrapper.displayName = 'NoMatchesFallbackWrapper';

type PickerProps<T extends object, U, C> = {
  /**
   * The options to display in the picker. This can be a flat array of options or an array of groups.
   */
  options: (T | Group<T>)[];
  /**
   * A function that returns the id of an option.
   */
  getOptionId: (option: T) => string;
  /**
   * A function that returns true if the option matches the search term.
   */
  isMatch: (option: T, searchTerm: string) => boolean;
  /**
   * A function that returns true if the option is disabled.
   */
  getIsDisabled?: (option: T) => boolean;
  /**
   * The currently selected item.
   */
  selectedItem?: T;
  /**
   * A function that is called when an option is selected.
   */
  onSelect?: (option: T) => void;
  /**
   * A function that is called when the picker is closed.
   */
  onClose?: () => void;
  /**
   * A ref to an imperative handle that can be used to control the picker.
   */
  handleRef?: React.Ref<ImperativeModelPickerHandle>;
  /**
   * A custom search bar component. If not provided, a default search bar will be used.
   */
  SearchBarComponent?: ReturnType<typeof fixedForwardRef<HTMLInputElement, InputProps>>;
  /**
   * A custom option component. If not provided, a default option component will be used.
   */
  OptionComponent?: React.ComponentType<
    {
      option: T;
    } & BoxProps
  >;
  /**
   * A custom group component. If not provided, a default group component will be used.
   */
  GroupComponent?: React.ComponentType<PropsWithChildren<{ group: Group<T, U> } & BoxProps>>;
  /**
   * A fallback component to display when there are no options. If a string is provided, it will be formatted
   * as a text element with appropriate styling. If a React node is provided, it will be rendered as is.
   */
  noOptionsFallback?: React.ReactNode;
  /**
   * A fallback component to display when there are no matches. If a string is provided, it will be formatted
   * as a text element with appropriate styling. If a React node is provided, it will be rendered as is.
   */
  noMatchesFallback?: React.ReactNode;
  /**
   * An optional object that can be used to pass additional data to custom picker components.
   */
  extra: C;
};

type PickerContextState<T extends object, U, C> = {
  options: (T | Group<T>)[];
  getOptionId: (option: T) => string;
  isMatch: (option: T, searchTerm: string) => boolean;
  getIsDisabled?: (option: T) => boolean;
  setActiveOptionId: (id: string) => void;
  onSelectById: (id: string) => void;
  setSearchTerm: (searchTerm: string) => void;
  SearchBarComponent: ReturnType<typeof fixedForwardRef<HTMLInputElement, InputProps>>;
  noOptionsFallback?: React.ReactNode;
  noMatchesFallback?: React.ReactNode;
  OptionComponent: React.ComponentType<{ option: T } & BoxProps>;
  GroupComponent: React.ComponentType<PropsWithChildren<{ group: Group<T, U> } & BoxProps>>;
  extra: C;
};

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const PickerContext = createContext<PickerContextState<any, any, any> | null>(null);
export const usePickerContext = <T extends object, U, C>(): PickerContextState<T, U, C> => {
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

export const Picker = typedMemo(<T extends object, U = undefined, C = undefined>(props: PickerProps<T, U, C>) => {
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
    noMatchesFallback,
    noOptionsFallback,
    OptionComponent = DefaultOptionComponent,
    GroupComponent = DefaultGroupComponent,
    extra,
  } = props;
  const [activeOptionId, setActiveOptionId, getActiveOptionId] = useStateImperative(() =>
    getFirstOptionId(options, getOptionId)
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [filteredOptions, setFilteredOptions] = useState<(T | Group<T, U>)[]>(options);
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
      const filtered: (T | Group<T, U>)[] = [];
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
        options,
        getOptionId,
        isMatch,
        getIsDisabled,
        onSelectById,
        setActiveOptionId,
        SearchBarComponent,
        noOptionsFallback,
        noMatchesFallback,
        OptionComponent,
        GroupComponent,
        extra,
        setSearchTerm,
      }) satisfies PickerContextState<T, U, C>,
    [
      options,
      getOptionId,
      isMatch,
      getIsDisabled,
      onSelectById,
      setActiveOptionId,
      SearchBarComponent,
      noOptionsFallback,
      noMatchesFallback,
      OptionComponent,
      GroupComponent,
      extra,
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
        <SearchBarComponent
          ref={inputRef}
          value={searchTerm}
          onChange={onChangeSearchTerm}
          isDisabled={flattenedOptions.length === 0}
        />
        <Flex tabIndex={-1} w="full" flexGrow={1}>
          {flattenedOptions.length === 0 && <NoOptionsFallbackWrapper>{noOptionsFallback}</NoOptionsFallbackWrapper>}
          {flattenedOptions.length > 0 && flattenedFilteredOptions.length === 0 && (
            <NoMatchesFallbackWrapper>{noMatchesFallback}</NoMatchesFallbackWrapper>
          )}
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
  <T extends object, U, C>({
    items,
    activeOptionId,
    selectedItemId,
  }: {
    items: (T | Group<T, U>)[];
    activeOptionId: string | undefined;
    selectedItemId: string | undefined;
  }) => {
    const { getOptionId, getIsDisabled } = usePickerContext<T, U, C>();

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
  <T extends object, U, C>({
    group,
    activeOptionId,
    selectedItemId,
  }: {
    group: Group<T, U>;
    activeOptionId: string | undefined;
    selectedItemId: string | undefined;
  }) => {
    const { getOptionId, GroupComponent, getIsDisabled } = usePickerContext<T, U, C>();

    return (
      <GroupComponent group={group}>
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
      </GroupComponent>
    );
  }
);
PickerOptionGroup.displayName = 'PickerOptionGroup';

const PickerOption = typedMemo(
  <T extends object, U, C>(props: {
    id: string;
    option: T;
    isActive: boolean;
    isSelected: boolean;
    isDisabled: boolean;
  }) => {
    const { OptionComponent, setActiveOptionId, onSelectById } = usePickerContext<T, U, C>();
    const { id, option, isActive, isDisabled, isSelected } = props;
    const onPointerMove = useCallback(() => {
      setActiveOptionId(id);
    }, [id, setActiveOptionId]);
    const onClick = useCallback(() => {
      onSelectById(id);
    }, [id, onSelectById]);
    return (
      <OptionComponent
        tabIndex={-1}
        option={option}
        id={id}
        data-disabled={isDisabled}
        data-selected={isSelected}
        data-active={isActive}
        onPointerMove={isDisabled ? undefined : onPointerMove}
        onClick={isDisabled ? undefined : onClick}
      />
    );
  }
);
PickerOption.displayName = 'PickerOption';
