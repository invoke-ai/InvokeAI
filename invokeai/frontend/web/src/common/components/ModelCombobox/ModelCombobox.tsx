import { Box, Flex, Input, Modal, ModalBody, ModalContent, ModalOverlay } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { ModelComboboxItem } from 'common/components/ModelCombobox/ModelComboboxItem';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useStateImperative } from 'common/hooks/useStateImperative';
import { atom } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig, BaseModelType } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export type ModelComboboxOptions = {
  modelConfigs: AnyModelConfig[];
  onSelect: (modelConfig: AnyModelConfig) => void;
  onClose?: () => void;
  noModelsInstalledFallback?: React.ReactNode;
  noModelsFoundFallback?: React.ReactNode;
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
    const [value, setValue, getValue] = useStateImperative(modelConfigs[0]?.key ?? '');
    const rootRef = useRef<HTMLDivElement>(null);
    const [items, setItems] = useState<AnyModelConfig[]>(modelConfigs);
    const [searchTerm, setSearchTerm] = useState('');
    const [debouncedSearchTerm] = useDebounce(searchTerm, 300);

    const onChangeSearchTerm = useCallback((e: ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    }, []);

    useEffect(() => {
      if (!debouncedSearchTerm) {
        setItems(modelConfigs);
        setValue(modelConfigs[0]?.key ?? '');
      } else {
        const lowercasedSearchTerm = debouncedSearchTerm.toLowerCase();
        const filtered = modelConfigs.filter((model) => isMatch(model, lowercasedSearchTerm));
        setItems(filtered);
        setValue(filtered[0]?.key ?? '');
      }
    }, [modelConfigs, debouncedSearchTerm, setValue]);

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
      [_onSelect, getValue, inputRef, items, next, prev]
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
                label="No matching models"
              />
            )}
            {items.length > 0 &&
              items.map((model) => (
                <ModelComboboxItem
                  key={model.key}
                  model={model}
                  setActive={setValue}
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
