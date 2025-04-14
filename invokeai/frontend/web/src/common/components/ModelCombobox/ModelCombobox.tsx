import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Input, Modal, ModalBody, ModalContent, ModalOverlay, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { atom } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

export type ModelComboboxOptions = {
  modelConfigs: AnyModelConfig[];
  onSelect: (modelConfig: AnyModelConfig) => void;
  onClose?: () => void;
};

const $modelComboboxState = atom<ModelComboboxOptions | null>(null);

const openModelCombobox = (options: ModelComboboxOptions) => {
  $modelComboboxState.set(options);
};

const closeModelCombobox = () => {
  $modelComboboxState.set(null);
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

const isMatch = (model: AnyModelConfig, searchTerm: string) => {
  const regex = getRegex(searchTerm);

  if (
    model.name.includes(searchTerm) ||
    regex.test(model.name) ||
    model.base.includes(searchTerm) ||
    regex.test(model.base) ||
    model.type.includes(searchTerm) ||
    regex.test(model.type) ||
    (model.description ?? '').includes(searchTerm) ||
    regex.test(model.description ?? '') ||
    model.format.includes(searchTerm) ||
    regex.test(model.format)
  ) {
    return true;
  }

  return false;
};

export const useModelCombobox = (options: ModelComboboxOptions) => {
  const onOpen = useCallback(() => {
    openModelCombobox(options);
  }, [options]);
  const onClose = useCallback(() => {
    closeModelCombobox();
  }, []);
  return {
    onOpen,
    onClose,
  };
};

export const ModelCombobox = memo(() => {
  const inputRef = useRef<HTMLInputElement>(null);
  const state = useStore($modelComboboxState);

  const onSelect = useCallback(
    (model: AnyModelConfig) => {
      if (!state) {
        // If the command menu is closed, we shouldn't do anything
        return;
      }
      state.onSelect(model);
      closeModelCombobox();
    },
    [state]
  );

  return (
    <Modal
      isOpen={!!state}
      onClose={closeModelCombobox}
      useInert={false}
      initialFocusRef={inputRef}
      size="xl"
      isCentered
    >
      <ModalOverlay />
      <ModalContent h="512" maxH="70%">
        <ModalBody p={0}>
          {state && <ModelComboboxContent inputRef={inputRef} modelConfigs={state.modelConfigs} onSelect={onSelect} />}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

ModelCombobox.displayName = 'ModelCombobox';

const ModelComboboxContent = memo(
  (props: {
    inputRef: RefObject<HTMLInputElement>;
    modelConfigs: AnyModelConfig[];
    onSelect: (model: AnyModelConfig) => void;
  }) => {
    const { inputRef, modelConfigs, onSelect: _onSelect } = props;
    const { t } = useTranslation();
    const [$value] = useState(() => atom(modelConfigs[0]?.key ?? ''));
    const value = useStore($value);
    const rootRef = useRef<HTMLDivElement>(null);
    // const [value, setValue] = useState(modelConfigs[0]?.key ?? '');
    const [items, setItems] = useState<AnyModelConfig[]>(modelConfigs);
    const [searchTerm, setSearchTerm] = useState('');

    const onChangeSearchTerm = useCallback(
      (e: ChangeEvent<HTMLInputElement>) => {
        setSearchTerm(e.target.value);
        if (!e.target.value) {
          setItems(modelConfigs);
          $value.set(modelConfigs[0]?.key ?? '');
        } else {
          const filtered = modelConfigs.filter((model) => isMatch(model, e.target.value));
          setItems(filtered);
          $value.set(filtered[0]?.key ?? '');
        }
      },
      [$value, modelConfigs]
    );

    const onSelect = useCallback(
      (key: string) => {
        const model = modelConfigs.find((model) => model.key === key);
        if (!model) {
          // Model not found? We should never get here.
          return;
        }
        _onSelect(model);
      },
      [_onSelect, modelConfigs]
    );

    const setValueAndScrollIntoView = useCallback(
      (key: string) => {
        $value.set(key);
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
      [$value]
    );

    const prev = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const value = $value.get();
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
      [$value, items, setValueAndScrollIntoView]
    );

    const next = useCallback(
      (e: React.KeyboardEvent) => {
        e.preventDefault();
        const value = $value.get();
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
      [$value, items, setValueAndScrollIntoView]
    );

    const onKeyDown = useCallback(
      (e: React.KeyboardEvent) => {
        if (e.key === 'ArrowUp') {
          prev(e);
        } else if (e.key === 'ArrowDown') {
          next(e);
        } else if (e.key === 'Enter') {
          const value = $value.get();
          const model = items.find((model) => model.key === value);
          if (!model) {
            // Model not found? We should never get here.
            return;
          }
          _onSelect(model);
          closeModelCombobox();
        } else if (e.key === 'Escape') {
          closeModelCombobox();
        } else if (e.key === '/') {
          e.preventDefault();
          inputRef.current?.focus();
          inputRef.current?.select();
        }
      },
      [$value, _onSelect, inputRef, items, next, prev]
    );

    return (
      <Flex tabIndex={-1} ref={rootRef} flexDir="column" p={2} h="full" gap={2} onKeyDown={onKeyDown}>
        <Input ref={inputRef} value={searchTerm} onChange={onChangeSearchTerm} placeholder={t('nodes.nodeSearch')} />
        <Box tabIndex={-1} role="listbox" w="full" h="full">
          <ScrollableContent>
            {items.length === 0 && (
              <IAINoContentFallback
                position="absolute"
                top={0}
                right={0}
                bottom={0}
                left={0}
                icon={null}
                label="No matching items"
              />
            )}
            {items.length > 0 &&
              items.map((model) => (
                <ModelComboboxItem
                  key={model.key}
                  model={model}
                  setActive={$value.set}
                  onSelect={onSelect}
                  isSelected={model.key === value}
                  isDisabled={false}
                />
              ))}
          </ScrollableContent>
        </Box>
      </Flex>
    );
  }
);
ModelComboboxContent.displayName = 'ModelComboboxContent';

const itemSx: SystemStyleObject = {
  display: 'flex',
  flexDir: 'column',
  py: 1,
  px: 2,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
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
    <>
      <Flex tabIndex={-1} gap={2} alignItems="center" justifyContent="space-between">
        <Text fontSize="sm" fontWeight="semibold">
          {model.name}
        </Text>
        <Text fontSize="sm" color="base.500">
          {model.base}
        </Text>
      </Flex>
      {model.description && <Text color="base.200">{model.description}</Text>}
    </>
  );
});
ModelComboboxItemContent.displayName = 'ModelComboboxItemContent';
