import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Divider, Flex, Input, Spacer, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { filesize } from 'filesize';
import type { ChangeEvent } from 'react';
import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';

export type ModelConfigGroup = {
  name: string;
  description: string;
  models: AnyModelConfig[];
};

const isModelConfigGroup = (modelConfig: AnyModelConfig | ModelConfigGroup): modelConfig is ModelConfigGroup => {
  return modelConfig ? 'models' in modelConfig : false;
};

export type ModelPickerProps = {
  modelConfigs: AnyModelConfig[] | ModelConfigGroup[];
  selectedModelConfig?: AnyModelConfig;
  onSelect?: (modelConfig: AnyModelConfig) => void;
  onClose?: () => void;
  noModelsInstalledFallback?: React.ReactNode;
  noModelsFoundFallback?: React.ReactNode;
};

export type ImperativeModelPickerHandle = {
  inputRef: React.RefObject<HTMLInputElement>;
  rootRef: React.RefObject<HTMLDivElement>;
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
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

const isMatch = (model: AnyModelConfig, searchTerm: string) => {
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

const getKeyOfFirstModel = (
  modelConfigs: (AnyModelConfig | ModelConfigGroup)[],
  selectedModelConfig?: AnyModelConfig
): string => {
  if (selectedModelConfig) {
    return selectedModelConfig.key;
  }
  const first = modelConfigs[0];
  if (!first) {
    return '';
  }
  if (isModelConfigGroup(first)) {
    return first.models[0]?.key ?? '';
  }
  return first?.key ?? '';
};

const findModel = (modelConfigs: (AnyModelConfig | ModelConfigGroup)[], key: string): AnyModelConfig | undefined => {
  for (const modelConfig of modelConfigs) {
    if (isModelConfigGroup(modelConfig)) {
      const model = modelConfig.models.find((model) => model.key === key);
      if (model) {
        return model;
      }
    } else {
      if (modelConfig.key === key) {
        return modelConfig;
      }
    }
  }
};

export const ModelPicker = memo(
  forwardRef<ImperativeModelPickerHandle, ModelPickerProps>((props, ref) => {
    const { t } = useTranslation();
    const [activeModelKey, setActiveModelKey, getActiveModelKey] = useStateImperative(() =>
      getKeyOfFirstModel(props.modelConfigs, props.selectedModelConfig)
    );
    const rootRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const [items, setItems] = useState<(AnyModelConfig | ModelConfigGroup)[]>(props.modelConfigs);
    const flatItems = useMemo(
      () => items.flatMap((item) => (isModelConfigGroup(item) ? item.models : [item])),
      [items]
    );
    const [searchTerm, setSearchTerm] = useState('');
    useImperativeHandle(ref, () => ({ inputRef, rootRef, searchTerm, setSearchTerm }), [searchTerm]);

    const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    }, []);

    useEffect(() => {
      if (!searchTerm) {
        setItems(props.modelConfigs);
        setActiveModelKey(getKeyOfFirstModel(props.modelConfigs));
      } else {
        const lowercasedSearchTerm = searchTerm.toLowerCase();
        const filtered: (AnyModelConfig | ModelConfigGroup)[] = [];
        for (const item of props.modelConfigs) {
          if (isModelConfigGroup(item)) {
            const filteredModels = item.models.filter((model) => isMatch(model, lowercasedSearchTerm));
            if (filteredModels.length > 0) {
              filtered.push({ ...item, models: filteredModels });
            }
          } else {
            if (isMatch(item, searchTerm)) {
              filtered.push(item);
            }
          }
        }
        setItems(filtered);
        setActiveModelKey(getKeyOfFirstModel(filtered));
      }
    }, [searchTerm, setActiveModelKey, props.modelConfigs]);

    const onSelect = useCallback(
      (key: string) => {
        const _onSelect = props.onSelect;
        const model = findModel(props.modelConfigs, key);
        if (!model) {
          // Model not found? We should never get here.
          return;
        }
        _onSelect?.(model);
      },
      [props.modelConfigs, props.onSelect]
    );

    const setValueAndScrollIntoView = useCallback(
      (key: string) => {
        setActiveModelKey(key);
        const rootEl = rootRef.current;
        if (!rootEl) {
          return;
        }
        const itemEl = rootEl.querySelector(`#${CSS.escape(key)}`);
        if (!itemEl) {
          return;
        }
        itemEl.scrollIntoView({ block: 'nearest' });
      },
      [setActiveModelKey]
    );

    const prev = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const activeModelKey = getActiveModelKey();
        if (flatItems.length === 0) {
          return;
        }
        if (e.metaKey) {
          const item = flatItems.at(0);
          if (item) {
            setValueAndScrollIntoView(item.key);
          }
          return;
        }
        const currentIndex = flatItems.findIndex((model) => model.key === activeModelKey);
        if (currentIndex < 0) {
          return;
        }
        let newIndex = currentIndex - 1;
        if (newIndex < 0) {
          newIndex = flatItems.length - 1;
        }
        const item = flatItems.at(newIndex);
        if (item) {
          setValueAndScrollIntoView(item.key);
        }
      },
      [getActiveModelKey, flatItems, setValueAndScrollIntoView]
    );

    const next = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const activeModelKey = getActiveModelKey();
        if (flatItems.length === 0) {
          return;
        }
        if (e.metaKey) {
          const item = flatItems.at(-1);
          if (item) {
            setValueAndScrollIntoView(item.key);
          }
          return;
        }

        const currentIndex = flatItems.findIndex((model) => model.key === activeModelKey);
        if (currentIndex < 0) {
          return;
        }
        let newIndex = currentIndex + 1;
        if (newIndex >= flatItems.length) {
          newIndex = 0;
        }
        const item = flatItems.at(newIndex);
        if (item) {
          setValueAndScrollIntoView(item.key);
        }
      },
      [getActiveModelKey, flatItems, setValueAndScrollIntoView]
    );

    const onKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowUp') {
          prev(e);
        } else if (e.key === 'ArrowDown') {
          next(e);
        } else if (e.key === 'Enter') {
          const activeModelKey = getActiveModelKey();
          const model = flatItems.find((model) => model.key === activeModelKey);
          if (!model) {
            // Model not found? We should never get here.
            return;
          }
          const _onSelect = props.onSelect;
          _onSelect?.(model);
        } else if (e.key === 'Escape') {
          const _onClose = props.onClose;
          _onClose?.();
        } else if (e.key === '/') {
          e.preventDefault();
          inputRef.current?.focus();
          inputRef.current?.select();
        }
      },
      [getActiveModelKey, flatItems, next, prev, props.onClose, props.onSelect]
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
          <ScrollableContent>
            <ModelPickerList
              items={items}
              activeModelKey={activeModelKey}
              setActiveModelKey={setActiveModelKey}
              selectedModelKey={props.selectedModelConfig?.key}
              onSelect={onSelect}
            />
          </ScrollableContent>
        </Flex>
      </Flex>
    );
  })
);
ModelPicker.displayName = 'ModelComboboxContent';

const ModelPickerList = memo(
  ({
    items,
    activeModelKey,
    setActiveModelKey,
    selectedModelKey,
    onSelect,
  }: {
    items: (AnyModelConfig | ModelConfigGroup)[];
    activeModelKey: string;
    selectedModelKey: string | undefined;
    setActiveModelKey: (key: string) => void;
    onSelect: (key: string) => void;
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
        {items.map((item) => {
          if (isModelConfigGroup(item)) {
            return (
              <Flex key={item.name} flexDir="column" gap={2} w="full">
                <Text fontSize="sm" fontWeight="semibold">
                  {item.name}
                </Text>
                <Text color="base.200" fontSize="xs">
                  {item.description}
                </Text>
                <Flex flexDir="column" gap={2} w="full">
                  {item.models.map((model) => (
                    <ModelPickerItem
                      key={model.key}
                      model={model}
                      setActive={setActiveModelKey}
                      onSelect={onSelect}
                      isActive={model.key === activeModelKey}
                      isSelected={model.key === selectedModelKey}
                      isDisabled={false}
                    />
                  ))}
                </Flex>
              </Flex>
            );
          } else {
            return (
              <ModelPickerItem
                key={item.key}
                model={item}
                setActive={setActiveModelKey}
                onSelect={onSelect}
                isActive={item.key === activeModelKey}
                isSelected={item.key === selectedModelKey}
                isDisabled={false}
              />
            );
          }
        })}
      </Flex>
    );
  }
);
ModelPickerList.displayName = 'ModelComboboxList';

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

const ModelPickerItem = memo(
  (props: {
    model: AnyModelConfig;
    setActive: (key: string) => void;
    onSelect: (key: string) => void;
    isActive: boolean;
    isSelected: boolean;
    isDisabled: boolean;
  }) => {
    const { model, setActive, onSelect, isActive, isDisabled, isSelected } = props;
    const onPointerMove = useCallback(() => {
      setActive(model.key);
    }, [model.key, setActive]);
    const onClick = useCallback(() => {
      onSelect(model.key);
    }, [model.key, onSelect]);
    return (
      <Box
        role="option"
        sx={itemSx}
        id={model.key}
        data-disabled={isDisabled}
        data-selected={isSelected}
        data-active={isActive}
        onPointerMove={isDisabled ? undefined : onPointerMove}
        onClick={isDisabled ? undefined : onClick}
      >
        <ModelPickerItemContent model={model} />
      </Box>
    );
  }
);
ModelPickerItem.displayName = 'ModelComboboxItem';

const ModelPickerItemContent = memo(({ model }: { model: AnyModelConfig }) => {
  return (
    <Flex tabIndex={-1} gap={2}>
      <ModelImage image_url={model.cover_image} />
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text fontSize="sm" fontWeight="semibold" noOfLines={1}>
            {model.name}
          </Text>
          <Spacer />
          <Text variant="subtext" fontStyle="italic" noOfLines={1} flexShrink={0} overflow="visible">
            {filesize(model.file_size)}
          </Text>
          <ModelBaseBadge base={model.base} />
        </Flex>
        {model.description && <Text color="base.200">{model.description}</Text>}
      </Flex>
    </Flex>
  );
});
ModelPickerItemContent.displayName = 'ModelComboboxItemContent';
