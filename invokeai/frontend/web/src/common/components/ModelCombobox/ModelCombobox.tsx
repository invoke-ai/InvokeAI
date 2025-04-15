import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  Flex,
  Input,
  Modal,
  ModalBody,
  ModalContent,
  ModalOverlay,
  Spacer,
  Text,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useStateImperative } from 'common/hooks/useStateImperative';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { toast } from 'features/toast/toast';
import { filesize } from 'filesize';
import type { ChangeEvent } from 'react';
import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAllModels } from 'services/api/hooks/modelsByType';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export type ModelPickerProps = {
  modelConfigs: AnyModelConfig[];
  onSelect?: (modelConfig: AnyModelConfig) => void;
  onClose?: () => void;
  noModelsInstalledFallback?: React.ReactNode;
  noModelsFoundFallback?: React.ReactNode;
};

export type ImperativeModelPickerHandle = {
  inputRef: React.RefObject<HTMLInputElement>;
  rootRef: React.RefObject<HTMLDivElement>;
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

const onSelect = (modelConfig: AnyModelConfig) => {
  // Handle model selection
  toast({
    description: `Selected model: ${modelConfig.name}`,
  });
};

export const ModelCombobox = memo(() => {
  const inputRef = useRef<HTMLInputElement>(null);
  const pickerRef = useRef<ImperativeModelPickerHandle>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [modelConfigs] = useAllModels();

  return (
    <>
      <Button onClick={onOpen} variant="outline">
        model
      </Button>
      <Modal isOpen={isOpen} onClose={onClose} useInert={false} initialFocusRef={inputRef} size="xl" isCentered>
        <ModalOverlay />
        <ModalContent h="512" maxH="70%">
          <ModalBody p={0}>
            <ModelComboboxContent ref={pickerRef} modelConfigs={modelConfigs} onSelect={onSelect} />
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
});

ModelCombobox.displayName = 'ModelCombobox';

const ModelComboboxContent = memo(
  forwardRef<ImperativeModelPickerHandle, ModelPickerProps>((props, ref) => {
    const { t } = useTranslation();
    const [value, setValue, getValue] = useStateImperative(props.modelConfigs[0]?.key ?? '');
    const rootRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    useImperativeHandle(ref, () => ({ inputRef, rootRef }), []);
    const [items, setItems] = useState<AnyModelConfig[]>(props.modelConfigs);
    const [searchTerm, setSearchTerm] = useState('');
    const [debouncedSearchTerm] = useDebounce(searchTerm, 300);

    const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    }, []);

    useEffect(() => {
      if (!debouncedSearchTerm) {
        setItems(props.modelConfigs);
        setValue(props.modelConfigs[0]?.key ?? '');
      } else {
        const lowercasedSearchTerm = debouncedSearchTerm.toLowerCase();
        const filtered = props.modelConfigs.filter((model) => isMatch(model, lowercasedSearchTerm));
        setItems(filtered);
        setValue(filtered[0]?.key ?? '');
      }
    }, [debouncedSearchTerm, setValue, props.modelConfigs]);

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
        setValue(key);
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
      [setValue]
    );

    const prev = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const value = getValue();
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
        const currentIndex = items.findIndex((model) => model.key === value);
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
      [getValue, items, setValueAndScrollIntoView]
    );

    const next = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const value = getValue();
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

        const currentIndex = items.findIndex((model) => model.key === value);
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
      [getValue, items, setValueAndScrollIntoView]
    );

    const onKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowUp') {
          prev(e);
        } else if (e.key === 'ArrowDown') {
          next(e);
        } else if (e.key === 'Enter') {
          const value = getValue();
          const model = items.find((model) => model.key === value);
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
      [getValue, items, next, prev, props.onClose, props.onSelect]
    );

    return (
      <Flex tabIndex={-1} ref={rootRef} flexDir="column" p={2} h="full" gap={2} onKeyDown={onKeyDown}>
        <Input ref={inputRef} value={searchTerm} onChange={onChangeSearchTerm} placeholder={t('nodes.nodeSearch')} />
        <Box tabIndex={-1} role="listbox" w="full" h="full">
          {/* <ScrollableContent> */}
          <ModelComboboxList items={items} value={value} setValue={setValue} onSelect={onSelect} />
          {/* </ScrollableContent> */}
        </Box>
      </Flex>
    );
  })
);
ModelComboboxContent.displayName = 'ModelComboboxContent';

const ModelComboboxList = memo(
  ({
    items,
    value,
    setValue,
    onSelect,
  }: {
    items: AnyModelConfig[];
    value: string;
    setValue: (key: string) => void;
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
      <>
        {items.map((model) => (
          <ModelComboboxItem
            key={model.key}
            model={model}
            setActive={setValue}
            onSelect={onSelect}
            isSelected={model.key === value}
            isDisabled={false}
          />
        ))}
      </>
    );
  }
);
ModelComboboxList.displayName = 'ModelComboboxList';

const itemSx: SystemStyleObject = {
  display: 'flex',
  flexDir: 'column',
  p: 2,
  cursor: 'pointer',
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
  },
  '&[data-disabled="true"]': {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
};

const ModelComboboxItem = memo(
  (props: {
    model: AnyModelConfig;
    setActive: (key: string) => void;
    onSelect: (key: string) => void;
    isSelected: boolean;
    isDisabled: boolean;
  }) => {
    const { model, setActive, onSelect, isDisabled, isSelected } = props;
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
        aria-disabled={isDisabled}
        aria-selected={isSelected}
        data-disabled={isDisabled}
        data-selected={isSelected}
        onPointerMove={isDisabled ? undefined : onPointerMove}
        onClick={isDisabled ? undefined : onClick}
      >
        <ModelComboboxItemContent model={model} />
      </Box>
    );
  }
);
ModelComboboxItem.displayName = 'ModelComboboxItem';

const ModelComboboxItemContent = memo(({ model }: { model: AnyModelConfig }) => {
  return (
    <Flex tabIndex={-1} gap={2}>
      <ModelImage image_url={model.cover_image} />
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text fontSize="sm" fontWeight="semibold">
            {model.name}
          </Text>
          <Spacer />
          <Text variant="subtext" fontStyle="italic">
            {filesize(model.file_size)}
          </Text>
          <ModelBaseBadge base={model.base} />
        </Flex>
        {model.description && <Text color="base.200">{model.description}</Text>}
      </Flex>
    </Flex>
  );
});
ModelComboboxItemContent.displayName = 'ModelComboboxItemContent';
