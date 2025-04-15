import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Divider, Flex, Input, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import { typedMemo } from 'common/util/typedMemo';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import type { ChangeEvent } from 'react';
import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export type OptionGroup<T extends object, U = any> = {
  id: string;
  data: U;
  options: T[];
};

const isGroup = <T extends object>(item: T | OptionGroup<T>): item is OptionGroup<T> => {
  return item ? 'options' in item && Array.isArray(item.options) : false;
};

export type ImperativeModelPickerHandle = {
  inputRef: React.RefObject<HTMLInputElement>;
  rootRef: React.RefObject<HTMLDivElement>;
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
};

export type PickerProps<T extends object> = {
  options: (T | OptionGroup<T>)[];
  getId: (item: T) => string;
  isMatch: (item: T, searchTerm: string) => boolean;
  getIsDisabled?: (item: T) => boolean;
  selectedItem?: T;
  onSelect?: (item: T) => void;
  onClose?: () => void;
  noOptionsFallback?: React.ReactNode;
  noMatchesFallback?: React.ReactNode;
  handleRef?: React.Ref<ImperativeModelPickerHandle>;
  ItemComponent: React.ComponentType<{ item: T }>;
  GroupHeaderComponent?: React.ComponentType<{ group: OptionGroup<T> }>;
};

const getRegex = (searchTerm: string) =>
  new RegExp(
    searchTerm
      .trim()
      .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
      .split(' ')
      .join('.*'),
    'gi'
  );

const BASE_KEYWORDS: { [key in BaseModelType]?: string[] } = {
  'sd-1': ['sd1', 'sd1.4', 'sd1.5', 'sd-1'],
  'sd-2': ['sd2', 'sd2.0', 'sd2.1', 'sd-2'],
  'sd-3': ['sd3', 'sd3.0', 'sd3.5', 'sd-3'],
};

export const isMatch = (model: AnyModelConfig, searchTerm: string) => {
  const regex = getRegex(searchTerm);

  if (
    model.name.toLowerCase().includes(searchTerm) ||
    regex.test(model.name) ||
    (BASE_KEYWORDS[model.base] ?? [model.base]).some((kw) => kw.toLowerCase().includes(searchTerm) || regex.test(kw)) ||
    model.type.toLowerCase().includes(searchTerm) ||
    regex.test(model.type) ||
    (model.description ?? '').toLowerCase().includes(searchTerm) ||
    regex.test(model.description ?? '') ||
    model.format.toLowerCase().includes(searchTerm) ||
    regex.test(model.format)
  ) {
    return true;
  }

  return false;
};

const getFirstOption = <T extends object>(options: (T | OptionGroup<T>)[]): T | undefined => {
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
  options: (T | OptionGroup<T>)[],
  getId: (item: T) => string
): string | undefined => {
  const firstOptionOrGroup = getFirstOption(options);
  if (firstOptionOrGroup) {
    return getId(firstOptionOrGroup);
  } else {
    return undefined;
  }
};

const findOption = <T extends object>(
  options: (T | OptionGroup<T>)[],
  id: string,
  getId: (item: T) => string
): T | undefined => {
  for (const optionOrGroup of options) {
    if (isGroup(optionOrGroup)) {
      const option = optionOrGroup.options.find((opt) => getId(opt) === id);
      if (option) {
        return option;
      }
    } else {
      if (getId(optionOrGroup) === id) {
        return optionOrGroup;
      }
    }
  }
};

const flattenOptions = <T extends object>(options: (T | OptionGroup<T>)[]): T[] => {
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
  const { t } = useTranslation();
  const {
    getId,
    options,
    handleRef,
    isMatch,
    getIsDisabled,
    noMatchesFallback,
    noOptionsFallback,
    onClose,
    onSelect,
    selectedItem,
    ItemComponent,
    GroupHeaderComponent,
  } = props;
  const [activeOptionId, setActiveOptionId, getActiveOptionId] = useStateImperative(() =>
    getFirstOptionId(options, getId)
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [filteredOptions, setFilteredOptions] = useState<(T | OptionGroup<T>)[]>(options);
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
      setActiveOptionId(getFirstOptionId(options, getId));
    } else {
      const lowercasedSearchTerm = searchTerm.toLowerCase();
      const filtered: (T | OptionGroup<T>)[] = [];
      for (const item of props.options) {
        if (isGroup(item)) {
          const filteredItems = item.options.filter(
            (item) => !getIsDisabled?.(item) && isMatch(item, lowercasedSearchTerm)
          );
          if (filteredItems.length > 0) {
            filtered.push({ ...item, options: filteredItems });
          }
        } else {
          if (!getIsDisabled?.(item) && isMatch(item, searchTerm)) {
            filtered.push(item);
          }
        }
      }
      setFilteredOptions(filtered);
      setActiveOptionId(getFirstOptionId(filtered, getId));
    }
  }, [searchTerm, setActiveOptionId, props.options, options, getId, isMatch, getIsDisabled]);

  const onSelectInternal = useCallback(
    (id: string) => {
      const item = findOption(options, id, getId);
      if (!item) {
        // Model not found? We should never get here.
        return;
      }
      onSelect?.(item);
    },
    [getId, options, onSelect]
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
          setValueAndScrollIntoView(getId(item));
        }
        return;
      }
      const currentIndex = flattenedFilteredOptions.findIndex((item) => getId(item) === activeOptionId);
      if (currentIndex < 0) {
        return;
      }
      let newIndex = currentIndex - 1;
      if (newIndex < 0) {
        newIndex = flattenedFilteredOptions.length - 1;
      }
      const item = flattenedFilteredOptions.at(newIndex);
      if (item) {
        setValueAndScrollIntoView(getId(item));
      }
    },
    [getActiveOptionId, flattenedFilteredOptions, setValueAndScrollIntoView, getId]
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
          setValueAndScrollIntoView(getId(item));
        }
        return;
      }

      const currentIndex = flattenedFilteredOptions.findIndex((item) => getId(item) === activeOptionId);
      if (currentIndex < 0) {
        return;
      }
      let newIndex = currentIndex + 1;
      if (newIndex >= flattenedFilteredOptions.length) {
        newIndex = 0;
      }
      const item = flattenedFilteredOptions.at(newIndex);
      if (item) {
        setValueAndScrollIntoView(getId(item));
      }
    },
    [getActiveOptionId, flattenedFilteredOptions, setValueAndScrollIntoView, getId]
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'ArrowUp') {
        prev(e);
      } else if (e.key === 'ArrowDown') {
        next(e);
      } else if (e.key === 'Enter') {
        const activeOptionId = getActiveOptionId();
        const item = flattenedFilteredOptions.find((item) => getId(item) === activeOptionId);
        if (!item) {
          // Model not found? We should never get here.
          return;
        }
        onSelect?.(item);
      } else if (e.key === 'Escape') {
        onClose?.();
      } else if (e.key === '/') {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    },
    [prev, next, getActiveOptionId, flattenedFilteredOptions, onSelect, getId, onClose]
  );

  return (
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
      <Flex gap={2} alignItems="center">
        <Input ref={inputRef} value={searchTerm} onChange={onChangeSearchTerm} placeholder={t('nodes.nodeSearch')} />
        <NavigateToModelManagerButton />
      </Flex>
      <Divider />
      <Flex tabIndex={-1} w="full" flexGrow={1}>
        {flattenedOptions.length === 0 && noOptionsFallback}
        {flattenedOptions.length > 0 && flattenedFilteredOptions.length === 0 && noMatchesFallback}
        {flattenedOptions.length > 0 && flattenedFilteredOptions.length > 0 && (
          <ScrollableContent>
            <PickerList
              items={filteredOptions}
              getId={getId}
              activeOptionId={activeOptionId}
              setActiveOptionId={setActiveOptionId}
              selectedItemId={selectedItem ? getId(selectedItem) : undefined}
              onSelect={onSelectInternal}
              getIsDisabled={getIsDisabled}
              ItemComponent={ItemComponent}
              GroupHeaderComponent={GroupHeaderComponent}
            />
          </ScrollableContent>
        )}
      </Flex>
    </Flex>
  );
});
Picker.displayName = 'Picker';

const PickerList = typedMemo(
  <T extends object>({
    items,
    activeOptionId,
    setActiveOptionId,
    selectedItemId,
    onSelect,
    getId,
    getIsDisabled,
    ItemComponent,
    GroupHeaderComponent,
  }: {
    items: (T | OptionGroup<T>)[];
    activeOptionId: string | undefined;
    setActiveOptionId: (key: string) => void;
    selectedItemId: string | undefined;
    onSelect: (key: string) => void;
    getId: (item: T) => string;
    getIsDisabled?: (item: T) => boolean;
    ItemComponent: React.ComponentType<{ item: T }>;
    GroupHeaderComponent?: React.ComponentType<{ group: OptionGroup<T> }>;
  }) => {
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
      <Flex flexDir="column" gap={2} w="full">
        {items.map((itemOrGroup) => {
          if (isGroup(itemOrGroup)) {
            return (
              <PickerOptionGroup
                key={itemOrGroup.id}
                group={itemOrGroup}
                setActiveOptionId={setActiveOptionId}
                activeOptionId={activeOptionId}
                getId={getId}
                onSelect={onSelect}
                selectedItemId={selectedItemId}
                ItemComponent={ItemComponent}
                getIsDisabled={getIsDisabled}
                GroupHeaderComponent={GroupHeaderComponent}
              />
            );
          } else {
            const id = getId(itemOrGroup);
            return (
              <PickerOption
                key={id}
                id={id}
                item={itemOrGroup}
                setActiveOptionId={setActiveOptionId}
                onSelect={onSelect}
                isActive={id === activeOptionId}
                isSelected={id === selectedItemId}
                isDisabled={getIsDisabled?.(itemOrGroup) ?? false}
                ItemComponent={ItemComponent}
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
    getId,
    setActiveOptionId,
    onSelect,
    activeOptionId,
    selectedItemId,
    getIsDisabled,
    ItemComponent,
    GroupHeaderComponent,
  }: {
    group: OptionGroup<T>;
    getId: (item: T) => string;
    setActiveOptionId: (key: string) => void;
    onSelect: (key: string) => void;
    activeOptionId: string | undefined;
    selectedItemId: string | undefined;
    getIsDisabled?: (item: T) => boolean;
    ItemComponent: React.ComponentType<{ item: T }>;
    GroupHeaderComponent?: React.ComponentType<{ group: OptionGroup<T> }>;
  }) => {
    return (
      <Flex key={group.id} flexDir="column" gap={2} w="full">
        {GroupHeaderComponent ? <GroupHeaderComponent group={group} /> : <Text fontWeight="bold">{group.id}</Text>}
        <Flex flexDir="column" gap={2} w="full">
          {group.options.map((item) => {
            const id = getId(item);
            return (
              <PickerOption
                key={id}
                id={id}
                item={item}
                setActiveOptionId={setActiveOptionId}
                onSelect={onSelect}
                isActive={id === activeOptionId}
                isSelected={id === selectedItemId}
                isDisabled={getIsDisabled?.(item) ?? false}
                ItemComponent={ItemComponent}
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
  <T extends object>(props: {
    id: string;
    item: T;
    setActiveOptionId: (key: string) => void;
    onSelect: (key: string) => void;
    isActive: boolean;
    isSelected: boolean;
    isDisabled: boolean;
    ItemComponent: React.ComponentType<{ item: T }>;
  }) => {
    const { id, item, ItemComponent, setActiveOptionId, onSelect, isActive, isDisabled, isSelected } = props;
    const onPointerMove = useCallback(() => {
      setActiveOptionId(id);
    }, [id, setActiveOptionId]);
    const onClick = useCallback(() => {
      onSelect(id);
    }, [id, onSelect]);
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
        <ItemComponent item={item} />
      </Box>
    );
  }
);
PickerOption.displayName = 'PickerOption';
