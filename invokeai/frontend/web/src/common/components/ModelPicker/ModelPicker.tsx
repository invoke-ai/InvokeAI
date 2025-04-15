import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Input, Spacer, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { NavigateToModelManagerButton } from 'features/parameters/components/MainModel/NavigateToModelManagerButton';
import { filesize } from 'filesize';
import type { ChangeEvent } from 'react';
import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';

export type ModelPickerProps = {
  modelConfigs: AnyModelConfig[];
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

export const ModelPicker = memo(
  forwardRef<ImperativeModelPickerHandle, ModelPickerProps>((props, ref) => {
    const { t } = useTranslation();
    const [activeModelKey, setActiveModelKey, getActiveModelKey] = useStateImperative(
      props.selectedModelConfig?.key ?? props.modelConfigs[0]?.key ?? ''
    );
    const rootRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const [items, setItems] = useState<AnyModelConfig[]>(props.modelConfigs);
    const [searchTerm, setSearchTerm] = useState('');
    useImperativeHandle(ref, () => ({ inputRef, rootRef, searchTerm, setSearchTerm }), [searchTerm]);

    const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    }, []);

    useEffect(() => {
      if (!searchTerm) {
        setItems(props.modelConfigs);
        setActiveModelKey(props.modelConfigs[0]?.key ?? '');
      } else {
        const lowercasedSearchTerm = searchTerm.toLowerCase();
        const filtered = props.modelConfigs.filter((model) => isMatch(model, lowercasedSearchTerm));
        setItems(filtered);
        setActiveModelKey(filtered[0]?.key ?? '');
      }
    }, [searchTerm, setActiveModelKey, props.modelConfigs]);

    const onSelect = useCallback(
      (key: string) => {
        const _onSelect = props.onSelect;
        const model = props.modelConfigs.find((model) => model.key === key);
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
        if (items.length === 0) {
          return;
        }
        if (e.metaKey) {
          const item = items.at(0);
          if (item) {
            setValueAndScrollIntoView(item.key);
          }
          return;
        }
        const currentIndex = items.findIndex((model) => model.key === activeModelKey);
        if (currentIndex < 0) {
          return;
        }
        let newIndex = currentIndex - 1;
        if (newIndex < 0) {
          newIndex = items.length - 1;
        }
        const item = items.at(newIndex);
        if (item) {
          setValueAndScrollIntoView(item.key);
        }
      },
      [getActiveModelKey, items, setValueAndScrollIntoView]
    );

    const next = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const activeModelKey = getActiveModelKey();
        if (items.length === 0) {
          return;
        }
        if (e.metaKey) {
          const item = items.at(-1);
          if (item) {
            setValueAndScrollIntoView(item.key);
          }
          return;
        }

        const currentIndex = items.findIndex((model) => model.key === activeModelKey);
        if (currentIndex < 0) {
          return;
        }
        let newIndex = currentIndex + 1;
        if (newIndex >= items.length) {
          newIndex = 0;
        }
        const item = items.at(newIndex);
        if (item) {
          setValueAndScrollIntoView(item.key);
        }
      },
      [getActiveModelKey, items, setValueAndScrollIntoView]
    );

    const onKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowUp') {
          prev(e);
        } else if (e.key === 'ArrowDown') {
          next(e);
        } else if (e.key === 'Enter') {
          const activeModelKey = getActiveModelKey();
          const model = items.find((model) => model.key === activeModelKey);
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
      [getActiveModelKey, items, next, prev, props.onClose, props.onSelect]
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
    items: AnyModelConfig[];
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
      <Flex flexDir="column" gap={2} w="full" h="full">
        {items.map((model) => (
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
