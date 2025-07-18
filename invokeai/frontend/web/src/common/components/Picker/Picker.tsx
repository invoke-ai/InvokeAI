import type { BoxProps, SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Badge,
  Divider,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { typedMemo } from 'common/util/typedMemo';
import { NO_DRAG_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';
import { selectModelPickerCompactViewStates, setModelPickerCompactView } from 'features/ui/store/uiSlice';
import type { AnyStore, ReadableAtom, Task, WritableAtom } from 'nanostores';
import { atom, computed } from 'nanostores';
import type { StoreValues } from 'nanostores/computed';
import type { ChangeEvent, MouseEventHandler, PropsWithChildren, RefObject } from 'react';
import React, {
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
import {
  PiArrowCounterClockwiseBold,
  PiArrowsInLineVerticalBold,
  PiArrowsOutLineVerticalBold,
  PiXBold,
} from 'react-icons/pi';
import { assert } from 'tsafe';
import { useDebounce } from 'use-debounce';

const NO_WHEEL_NO_DRAG_CLASS = `${NO_WHEEL_CLASS} ${NO_DRAG_CLASS}`;

const uniqueGroupKey = Symbol('uniqueGroupKey');

export type Group<T extends object> = {
  /**
   * The unique id of the group.
   */
  id: string;
  /**
   * The options in the group.
   */
  options: T[];
  /**
   * The color of the group. Used to style the group toggle button and vertical group line.
   *
   * It can be a CSS color string or theme color token.
   */
  color?: string;
  /**
   * The name of the group.
   */
  name?: string;
  /**
   * The short name of the group. Used to display for the group toggle button.
   */
  shortName?: string;
  /**
   * The description of the group. Used to display in the group toggle button.
   */
  description?: string;
  /**
   * A function that returns a "count" string for the group. It will be called with the number of matching options in
   * the group.
   */
  getOptionCountString?: (count: number) => string;
  /**
   * A unique key used for type-checking the group. Use the `buildGroup` function to create a group, which will set this key.
   */
  [uniqueGroupKey]: true;
};

type OptionOrGroup<T extends object> = T | Group<T>;

export const buildGroup = <T extends object>(group: Omit<Group<T>, typeof uniqueGroupKey>): Group<T> => ({
  ...group,
  [uniqueGroupKey]: true,
});

export const isGroup = <T extends object>(optionOrGroup: OptionOrGroup<T>): optionOrGroup is Group<T> => {
  return uniqueGroupKey in optionOrGroup && optionOrGroup[uniqueGroupKey] === true;
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

type PickerProps<T extends object> = {
  /**
   * Unique identifier for this picker instance. Used to persist compact view state.
   */
  pickerId?: string;
  /**
   * The options to display in the picker. This can be a flat array of options or an array of groups.
   */
  optionsOrGroups: OptionOrGroup<T>[];
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
  getIsOptionDisabled?: (option: T) => boolean;
  /**
   * The currently selected item.
   */
  selectedOption?: T;
  /**
   * A function that is called when an option is selected.
   */
  onSelect?: (option: T) => void;
  /**
   * A function that is called when the picker is closed.
   */
  onClose?: () => void;
  /**
   * A placeholder for the search input.
   */
  searchPlaceholder?: string;
  /**
   * A ref to an imperative handle that can be used to control the picker.
   */
  handleRef?: React.Ref<PickerContextState<T>>;
  /**
   * A custom option component. If not provided, a default option component will be used.
   */
  OptionComponent?: React.ComponentType<{ option: T } & BoxProps>;
  /**
   * A component to render next to the search bar.
   */
  NextToSearchBar?: React.ReactNode;
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
   * Whether the picker should be searchable. If true, renders a search input.
   */
  searchable?: boolean;
  /**
   * Initial state for group toggles. If provided, groups will start with these states instead of all being disabled.
   */
  initialGroupStates?: GroupStatusMap;
};

export type PickerContextState<T extends object> = {
  $optionsOrGroups: WritableAtom<OptionOrGroup<T>[]>;
  $groupStatusMap: WritableAtom<GroupStatusMap>;
  isCompactView: boolean;
  $activeOptionId: WritableAtom<string | undefined>;
  $filteredOptions: WritableAtom<OptionOrGroup<T>[]>;
  $flattenedFilteredOptions: ReadableAtom<T[]>;
  $totalOptionCount: ReadableAtom<number>;
  $hasOptions: ReadableAtom<boolean>;
  $filteredOptionsCount: ReadableAtom<number>;
  $hasFilteredOptions: ReadableAtom<boolean>;
  $areAllGroupsDisabled: ReadableAtom<boolean>;
  $selectedItem: WritableAtom<T | undefined>;
  $selectedItemId: ReadableAtom<string | undefined>;
  $searchTerm: WritableAtom<string>;
  searchPlaceholder?: string;
  toggleGroup: (id: string) => void;
  getOptionId: (option: T) => string;
  isMatch: (option: T, searchTerm: string) => boolean;
  getIsOptionDisabled?: (option: T) => boolean;
  onSelectById: (id: string) => void;
  onClose?: () => void;
  rootRef: RefObject<HTMLDivElement>;
  inputRef: RefObject<HTMLInputElement>;
  noOptionsFallback?: React.ReactNode;
  noMatchesFallback?: React.ReactNode;
  OptionComponent: React.ComponentType<{ option: T } & BoxProps>;
  NextToSearchBar?: React.ReactNode;
  searchable?: boolean;
  pickerId?: string;
};

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const PickerContext = createContext<PickerContextState<any> | null>(null);
export const usePickerContext = <T extends object>(): PickerContextState<T> => {
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

const getFirstOption = <T extends object>(options: OptionOrGroup<T>[]): T | undefined => {
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
  options: OptionOrGroup<T>[],
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
  options: OptionOrGroup<T>[],
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

const flattenOptions = <T extends object>(options: OptionOrGroup<T>[]): T[] => {
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

export type GroupStatusMap = Record<string, boolean>;

const useTogglableGroups = <T extends object>(options: OptionOrGroup<T>[], initialGroupStates?: GroupStatusMap) => {
  const groupsWithOptions = useMemo(() => {
    const ids: string[] = [];
    for (const optionOrGroup of options) {
      if (isGroup(optionOrGroup) && !ids.includes(optionOrGroup.id)) {
        ids.push(optionOrGroup.id);
      }
    }
    return ids;
  }, [options]);

  const [$groupStatusMap] = useState(atom<GroupStatusMap>({}));
  const [$areAllGroupsDisabled] = useState(() =>
    computed($groupStatusMap, (groupStatusMap) => Object.values(groupStatusMap).every((status) => status === false))
  );

  useEffect(() => {
    const groupStatusMap = $groupStatusMap.get();
    const newMap: GroupStatusMap = {};
    for (const id of groupsWithOptions) {
      if (initialGroupStates && initialGroupStates[id] !== undefined) {
        newMap[id] = initialGroupStates[id];
      } else if (groupStatusMap[id] !== undefined) {
        newMap[id] = groupStatusMap[id];
      } else {
        newMap[id] = false;
      }
    }
    $groupStatusMap.set(newMap);
  }, [groupsWithOptions, $groupStatusMap, initialGroupStates]);

  const toggleGroup = useCallback(
    (idToToggle: string) => {
      const groupStatusMap = $groupStatusMap.get();
      const newMap: GroupStatusMap = {};
      for (const id of groupsWithOptions) {
        const prevStatus = Boolean(groupStatusMap[id]);
        newMap[id] = id === idToToggle ? !prevStatus : prevStatus;
      }
      $groupStatusMap.set(newMap);
    },
    [$groupStatusMap, groupsWithOptions]
  );

  return { $groupStatusMap, $areAllGroupsDisabled, toggleGroup } as const;
};

const useKeyboardNavigation = <T extends object>() => {
  const { getOptionId, $activeOptionId, $flattenedFilteredOptions, onSelectById, rootRef, onClose, inputRef } =
    usePickerContext<T>();

  const setValueAndScrollIntoView = useCallback(
    (id: string) => {
      $activeOptionId.set(id);
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
    [$activeOptionId, rootRef]
  );

  const prev = useCallback(
    (e: React.KeyboardEvent) => {
      e.preventDefault();
      const flattenedFilteredOptions = $flattenedFilteredOptions.get();
      const activeOptionId = $activeOptionId.get();
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
    [$activeOptionId, $flattenedFilteredOptions, setValueAndScrollIntoView, getOptionId]
  );

  const next = useCallback(
    (e: React.KeyboardEvent) => {
      e.preventDefault();
      const activeOptionId = $activeOptionId.get();
      const flattenedFilteredOptions = $flattenedFilteredOptions.get();
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
    [$activeOptionId, $flattenedFilteredOptions, setValueAndScrollIntoView, getOptionId]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowUp') {
        prev(e);
      } else if (e.key === 'ArrowDown') {
        next(e);
      } else if (e.key === 'Enter') {
        const activeOptionId = $activeOptionId.get();
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
    [prev, next, $activeOptionId, onSelectById, onClose, inputRef]
  );

  const keyboardNavProps = useMemo(() => {
    return {
      onKeyDown,
    };
  }, [onKeyDown]);

  return keyboardNavProps;
};

const useAtom = <T,>(initialValue: T) => {
  return useState(() => atom<T>(initialValue))[0];
};

const useComputed = <Value, OriginStores extends AnyStore[]>(
  stores: [...OriginStores],
  cb: (...values: StoreValues<OriginStores>) => Task<Value> | Value
) => {
  return useState(() => computed(stores, cb))[0];
};

const countOptions = <T extends object>(optionsOrGroups: OptionOrGroup<T>[]) => {
  let count = 0;
  for (const optionOrGroup of optionsOrGroups) {
    if (isGroup(optionOrGroup)) {
      count += optionOrGroup.options.length;
    } else {
      count++;
    }
  }
  return count;
};

export const Picker = typedMemo(<T extends object>(props: PickerProps<T>) => {
  const {
    pickerId,
    getOptionId,
    optionsOrGroups,
    handleRef,
    isMatch,
    getIsOptionDisabled,
    onClose,
    onSelect,
    selectedOption,
    searchPlaceholder,
    noMatchesFallback,
    noOptionsFallback,
    OptionComponent = DefaultOptionComponent,
    NextToSearchBar,
    searchable,
    initialGroupStates,
  } = props;
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const compactViewStates = useAppSelector(selectModelPickerCompactViewStates);

  const { $groupStatusMap, $areAllGroupsDisabled, toggleGroup } = useTogglableGroups(
    optionsOrGroups,
    initialGroupStates
  );
  const $activeOptionId = useAtom(getFirstOptionId(optionsOrGroups, getOptionId));
  const $optionsOrGroups = useAtom(optionsOrGroups);
  const $totalOptionCount = useComputed([$optionsOrGroups], countOptions);
  const $filteredOptions = useAtom<OptionOrGroup<T>[]>([]);
  const $flattenedFilteredOptions = useComputed([$filteredOptions], flattenOptions);
  const $hasOptions = useComputed([$totalOptionCount], (count) => count > 0);
  const $filteredOptionsCount = useComputed([$flattenedFilteredOptions], (options) => options.length);
  const $hasFilteredOptions = useComputed([$filteredOptionsCount], (count) => count > 0);
  const $selectedItem = useAtom<T | undefined>(undefined);
  const $searchTerm = useAtom('');
  const $selectedItemId = useComputed([$selectedItem], (item) => (item ? getOptionId(item) : undefined));

  // Use Redux state for compact view, defaulting to true if no pickerId or no saved state
  const isCompactView = pickerId ? (compactViewStates[pickerId] ?? true) : true;

  const onSelectById = useCallback(
    (id: string) => {
      const options = $filteredOptions.get();
      const item = findOption(options, id, getOptionId);
      if (!item) {
        // Model not found? We should never get here.
        return;
      }
      onSelect?.(item);
    },
    [$filteredOptions, getOptionId, onSelect]
  );

  // Sync the picker's nanostores when props change
  useEffect(() => {
    $selectedItem.set(selectedOption);
  }, [$selectedItem, selectedOption]);

  useEffect(() => {
    $optionsOrGroups.set(optionsOrGroups);
  }, [optionsOrGroups, $optionsOrGroups]);

  const ctx = useMemo(
    () =>
      ({
        $optionsOrGroups,
        $groupStatusMap,
        isCompactView,
        $activeOptionId,
        $filteredOptions,
        $flattenedFilteredOptions,
        $totalOptionCount,
        $selectedItem,
        $searchTerm,
        getOptionId,
        isMatch,
        getIsOptionDisabled,
        onSelectById,
        noOptionsFallback,
        noMatchesFallback,
        toggleGroup,
        rootRef,
        inputRef,
        searchPlaceholder,
        OptionComponent,
        NextToSearchBar,
        onClose,
        searchable,
        $areAllGroupsDisabled,
        $selectedItemId,
        $hasOptions,
        $hasFilteredOptions,
        $filteredOptionsCount,
        pickerId,
      }) satisfies PickerContextState<T>,
    [
      $optionsOrGroups,
      $groupStatusMap,
      isCompactView,
      $activeOptionId,
      $filteredOptions,
      $flattenedFilteredOptions,
      $totalOptionCount,
      $selectedItem,
      $searchTerm,
      getOptionId,
      isMatch,
      getIsOptionDisabled,
      onSelectById,
      noOptionsFallback,
      noMatchesFallback,
      toggleGroup,
      searchPlaceholder,
      OptionComponent,
      NextToSearchBar,
      onClose,
      searchable,
      $areAllGroupsDisabled,
      $selectedItemId,
      $hasOptions,
      $hasFilteredOptions,
      $filteredOptionsCount,
      pickerId,
    ]
  );

  useImperativeHandle(handleRef, () => ctx, [ctx]);

  return (
    <PickerContext.Provider value={ctx}>
      <PickerContainer>
        <PickerSearchBar />
        <Flex tabIndex={-1} w="full" flexGrow={1}>
          <NoOptionsFallback />
          <NoMatchesFallback />
          <PickerList />
        </Flex>
      </PickerContainer>
      <PickerSyncer />
    </PickerContext.Provider>
  );
});
Picker.displayName = 'Picker';

const PickerSyncer = typedMemo(<T extends object>() => {
  const {
    $optionsOrGroups,
    $searchTerm,
    $activeOptionId,
    $groupStatusMap,
    $areAllGroupsDisabled,
    $filteredOptions,
    searchable,
    isMatch,
    getOptionId,
  } = usePickerContext<T>();
  const searchTerm = useStore($searchTerm);
  const groupStatusMap = useStore($groupStatusMap);
  const areAllGroupsDisabled = useStore($areAllGroupsDisabled);
  const optionsOrGroups = useStore($optionsOrGroups);
  const [debouncedSearchTerm] = useDebounce(searchTerm, 300);

  useEffect(() => {
    if (!debouncedSearchTerm || !searchable) {
      const filtered = optionsOrGroups.filter((item) => {
        if (isGroup(item)) {
          return groupStatusMap[item.id] || areAllGroupsDisabled;
        } else {
          return true;
        }
      });
      $filteredOptions.set(filtered);
      $activeOptionId.set(getFirstOptionId(filtered, getOptionId));
    } else {
      const lowercasedSearchTerm = debouncedSearchTerm.toLowerCase();
      const filtered = [];
      for (const item of optionsOrGroups) {
        if (isGroup(item)) {
          if (!groupStatusMap[item.id] && !areAllGroupsDisabled) {
            continue;
          }
          const filteredItems = item.options.filter((item) => isMatch(item, lowercasedSearchTerm));
          if (filteredItems.length > 0) {
            filtered.push({ ...item, options: filteredItems });
          }
        } else {
          if (isMatch(item, debouncedSearchTerm)) {
            filtered.push(item);
          }
        }
      }
      $filteredOptions.set(filtered);
      $activeOptionId.set(getFirstOptionId(filtered, getOptionId));
    }
  }, [
    debouncedSearchTerm,
    $activeOptionId,
    getOptionId,
    isMatch,
    $filteredOptions,
    searchable,
    optionsOrGroups,
    groupStatusMap,
    areAllGroupsDisabled,
  ]);

  return null;
});
PickerSyncer.displayName = 'PickerSyncer';

const PickerContainer = typedMemo(({ children }: PropsWithChildren) => {
  const { rootRef } = usePickerContext();
  const keyboardNavProps = useKeyboardNavigation();
  return (
    <Flex
      className={NO_WHEEL_NO_DRAG_CLASS}
      tabIndex={-1}
      ref={rootRef}
      flexGrow={1}
      flexDir="column"
      p={2}
      w="full"
      h="full"
      gap={2}
      {...keyboardNavProps}
    >
      {children}
    </Flex>
  );
});
PickerContainer.displayName = 'PickerContainer';

const NoOptionsFallback = typedMemo(<T extends object>() => {
  const { noOptionsFallback, $hasOptions } = usePickerContext<T>();
  const hasOptions = useStore($hasOptions);

  if (hasOptions) {
    return null;
  }

  return <NoOptionsFallbackWrapper>{noOptionsFallback}</NoOptionsFallbackWrapper>;
});
NoOptionsFallback.displayName = 'NoOptionsFallback';

const NoMatchesFallback = typedMemo(<T extends object>() => {
  const { noMatchesFallback, $hasOptions, $hasFilteredOptions } = usePickerContext<T>();

  const hasOptions = useStore($hasOptions);
  const hasFilteredOptions = useStore($hasFilteredOptions);

  if (!hasOptions) {
    return null;
  }

  if (hasFilteredOptions) {
    return null;
  }

  return <NoMatchesFallbackWrapper>{noMatchesFallback}</NoMatchesFallbackWrapper>;
});
NoMatchesFallback.displayName = 'NoMatchesFallback';

const PickerSearchBar = typedMemo(<T extends object>() => {
  const { NextToSearchBar, searchable } = usePickerContext<T>();

  if (!searchable) {
    return null;
  }

  return (
    <Flex flexDir="column" w="full" gap={2}>
      <Flex gap={2} alignItems="center">
        <SearchInput />
        {NextToSearchBar}
        <CompactViewToggleButton />
      </Flex>
      <GroupToggleButtons />
    </Flex>
  );
});
PickerSearchBar.displayName = 'PickerSearchBar';

const SearchInput = typedMemo(<T extends object>() => {
  const { inputRef, $totalOptionCount, $searchTerm, searchPlaceholder } = usePickerContext<T>();
  const { t } = useTranslation();
  const searchTerm = useStore($searchTerm);
  const totalOptionCount = useStore($totalOptionCount);
  const placeholder = searchPlaceholder ?? t('common.search');
  const resetSearchTerm = useCallback(() => {
    $searchTerm.set('');
    inputRef.current?.focus();
  }, [$searchTerm, inputRef]);

  const onChangeSearchTerm = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      $searchTerm.set(e.target.value);
    },
    [$searchTerm]
  );
  return (
    <InputGroup>
      <Input ref={inputRef} value={searchTerm} onChange={onChangeSearchTerm} placeholder={placeholder} />
      {searchTerm && (
        <InputRightElement h="full" pe={2}>
          <IconButton
            onClick={resetSearchTerm}
            size="sm"
            variant="link"
            aria-label={t('common.clear')}
            tooltip={t('common.clear')}
            icon={<PiXBold />}
            isDisabled={totalOptionCount === 0}
            disabled={false}
          />
        </InputRightElement>
      )}
    </InputGroup>
  );
});
SearchInput.displayName = 'SearchInput';
const GroupToggleButtons = typedMemo(<T extends object>() => {
  const { $optionsOrGroups, $groupStatusMap, $areAllGroupsDisabled } = usePickerContext<T>();
  const { t } = useTranslation();
  const $groups = useComputed([$optionsOrGroups], (optionsOrGroups) => {
    const _groups: Group<T>[] = [];
    for (const optionOrGroup of optionsOrGroups) {
      if (isGroup(optionOrGroup)) {
        _groups.push(optionOrGroup);
      }
    }
    return _groups;
  });
  const groups = useStore($groups);
  const areAllGroupsDisabled = useStore($areAllGroupsDisabled);

  const onClick = useCallback<MouseEventHandler>(() => {
    const newMap: GroupStatusMap = {};
    for (const { id } of groups) {
      newMap[id] = false;
    }
    $groupStatusMap.set(newMap);
  }, [$groupStatusMap, groups]);

  if (!groups.length) {
    return null;
  }

  return (
    <Flex gap={2} alignItems="center">
      {groups.map((group) => (
        <GroupToggleButton key={group.id} group={group} />
      ))}
      <Spacer />
      <IconButton
        icon={<PiArrowCounterClockwiseBold />}
        aria-label={t('common.reset')}
        tooltip={t('common.reset')}
        size="sm"
        variant="link"
        alignSelf="stretch"
        onClick={onClick}
        // When a focused element is disabled, it blurs. This closes the popover. Fake the disabled state to prevent this.
        // See: https://github.com/chakra-ui/chakra-ui/issues/7965
        opacity={areAllGroupsDisabled ? 0.5 : undefined}
        pointerEvents={areAllGroupsDisabled ? 'none' : undefined}
      />
    </Flex>
  );
});
GroupToggleButtons.displayName = 'GroupToggleButtons';

const CompactViewToggleButton = typedMemo(<T extends object>() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { isCompactView, pickerId } = usePickerContext<T>();

  const onClick = useCallback(() => {
    if (pickerId) {
      dispatch(setModelPickerCompactView({ pickerId, isCompact: !isCompactView }));
    }
  }, [dispatch, pickerId, isCompactView]);

  const label = isCompactView ? t('common.fullView') : t('common.compactView');
  const icon = isCompactView ? <PiArrowsOutLineVerticalBold /> : <PiArrowsInLineVerticalBold />;

  return <IconButton aria-label={label} tooltip={label} size="sm" variant="ghost" icon={icon} onClick={onClick} />;
});
CompactViewToggleButton.displayName = 'CompactViewToggleButton';

const GroupToggleButton = typedMemo(<T extends object>({ group }: { group: Group<T> }) => {
  const { toggleGroup, $groupStatusMap } = usePickerContext<T>();
  const groupStatusMap = useStore($groupStatusMap);

  const onClick = useCallback(() => {
    toggleGroup(group.id);
  }, [group.id, toggleGroup]);

  const groupColor = getGroupColor(group);
  const shortName = getGroupShortName(group);
  const bg = groupStatusMap[group.id] ? groupColor : 'transparent';
  const color = groupStatusMap[group.id] ? undefined : 'base.200';

  return (
    <Badge
      role="button"
      size="xs"
      variant="solid"
      userSelect="none"
      bg={bg}
      color={color}
      borderColor={groupColor}
      borderWidth={1}
      onClick={onClick}
    >
      {shortName}
    </Badge>
  );
});
GroupToggleButton.displayName = 'GroupToggleButton';

const listSx = {
  flexDir: 'column',
  w: 'full',
  gap: 2,
  '&[data-is-compact="true"]': {
    gap: 1,
  },
} satisfies SystemStyleObject;

const PickerList = typedMemo(<T extends object>() => {
  const { getOptionId, isCompactView, $filteredOptions } = usePickerContext<T>();
  const compactView = isCompactView;
  const filteredOptions = useStore($filteredOptions);

  if (filteredOptions.length === 0) {
    return null;
  }

  return (
    <ScrollableContent>
      <Flex sx={listSx} data-is-compact={compactView}>
        {filteredOptions.map((optionOrGroup, i) => {
          if (isGroup(optionOrGroup)) {
            const withDivider = !compactView && i < filteredOptions.length - 1;
            return (
              <React.Fragment key={optionOrGroup.id}>
                <PickerGroup group={optionOrGroup} />
                {withDivider && <Divider />}
              </React.Fragment>
            );
          } else {
            const id = getOptionId(optionOrGroup);
            return <PickerOption id={id} key={id} option={optionOrGroup} />;
          }
        })}
      </Flex>
    </ScrollableContent>
  );
});
PickerList.displayName = 'PickerList';

const PickerGroup = typedMemo(<T extends object>({ group }: { group: Group<T> }) => {
  const { getOptionId, $groupStatusMap, $areAllGroupsDisabled } = usePickerContext<T>();

  const [$isGroupDisabled] = useState(() =>
    computed(
      [$groupStatusMap, $areAllGroupsDisabled],
      (groupStatusMap, areAllGroupsDisabled) => !groupStatusMap[group.id] && !areAllGroupsDisabled
    )
  );
  const isGroupDisabled = useStore($isGroupDisabled);

  if (isGroupDisabled) {
    return null;
  }

  return (
    <PickerGroupContainer group={group}>
      {group.options.map((item) => {
        const id = getOptionId(item);
        return <PickerOption key={id} id={id} option={item} />;
      })}
    </PickerGroupContainer>
  );
});
PickerGroup.displayName = 'PickerGroup';

const PickerOption = typedMemo(<T extends object>(props: { id: string; option: T }) => {
  const { OptionComponent, $activeOptionId, $selectedItemId, onSelectById, getIsOptionDisabled } =
    usePickerContext<T>();
  const { id, option } = props;
  const [$isActive] = useState(() => computed($activeOptionId, (activeOptionId) => activeOptionId === id));
  const [$isSelected] = useState(() => computed($selectedItemId, (selectedItemId) => selectedItemId === id));
  const isActive = useStore($isActive);
  const isSelected = useStore($isSelected);
  const setAsActive = useCallback(() => {
    $activeOptionId.set(id);
  }, [$activeOptionId, id]);
  const select = useCallback(() => {
    onSelectById(id);
  }, [id, onSelectById]);

  const isDisabled = getIsOptionDisabled?.(option) ?? false;
  const onPointerMove = isDisabled ? undefined : setAsActive;
  const onClick = isDisabled ? undefined : select;
  return (
    <OptionComponent
      tabIndex={-1}
      option={option}
      id={id}
      data-disabled={isDisabled}
      data-selected={isSelected}
      data-active={isActive}
      onPointerMove={onPointerMove}
      onClick={onClick}
    />
  );
});
PickerOption.displayName = 'PickerOption';

const getGroupColor = <T extends object>(group: Group<T>) => {
  return group.color ?? 'base.300';
};

const getGroupShortName = <T extends object>(group: Group<T>) => {
  return group.shortName ?? group.name ?? group.id;
};

const getGroupName = <T extends object>(group: Group<T>) => {
  return group.name ?? group.id;
};

const getGroupCount = <T extends object>(group: Group<T>, t: ReturnType<typeof useTranslation>['t']) => {
  return (
    group.getOptionCountString?.(group.options.length) ?? t('common.options_withCount', { count: group.options.length })
  );
};

const groupContainerSx = {
  flexDir: 'column',
  w: 'full',
  borderLeftWidth: 4,
  ps: 2,
  '&[data-all-disabled="true"]': {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
} satisfies SystemStyleObject;

const PickerGroupContainer = typedMemo(
  <T extends object>({ group, children }: PropsWithChildren<{ group: Group<T> }>) => {
    const { getIsOptionDisabled } = usePickerContext<T>();
    const color = getGroupColor(group);
    const areAllDisabled = group.options.every((item) => getIsOptionDisabled?.(item) ?? false);

    return (
      <Flex sx={groupContainerSx} borderLeftColor={color} data-all-disabled={areAllDisabled}>
        <PickerGroupHeader group={group} />
        <Flex flexDir="column" gap={1} w="full">
          {children}
        </Flex>
      </Flex>
    );
  }
);
PickerGroupContainer.displayName = 'PickerGroupContainer';

const groupHeaderSx = {
  flexDir: 'column',
  flex: 1,
  ps: 2,
  pe: 4,
  py: 1,
  userSelect: 'none',
  position: 'sticky',
  top: 0,
  bg: 'base.800',
  minH: 8,
  '&[data-is-compact="true"]': {
    ps: 1,
  },
} satisfies SystemStyleObject;

const PickerGroupHeader = typedMemo(<T extends object>({ group }: { group: Group<T> }) => {
  const { t } = useTranslation();
  const { isCompactView } = usePickerContext<T>();
  const compactView = isCompactView;
  const color = getGroupColor(group);
  const name = getGroupName(group);
  const count = getGroupCount(group, t);

  return (
    <Flex sx={groupHeaderSx} data-is-compact={compactView}>
      <Flex gap={2} alignItems="center">
        <Text fontSize="sm" fontWeight="semibold" color={color} noOfLines={1}>
          {name}
        </Text>
        <Spacer />
        <Text fontSize="sm" color="base.300" noOfLines={1}>
          {count}
        </Text>
      </Flex>
    </Flex>
  );
});
PickerGroupHeader.displayName = 'PickerGroupHeader';
