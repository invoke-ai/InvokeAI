import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  chakra,
  Flex,
  Input,
  Modal,
  ModalBody,
  ModalContent,
  ModalOverlay,
  Spacer,
  Text,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { EMPTY_ARRAY } from 'app/store/constants';
import { CommandEmpty, CommandItem, CommandList, CommandRoot } from 'cmdk';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { LRUCache } from 'lru-cache';
import { atom } from 'nanostores';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export type ModelCmdkOptions = {
  filter?: (modelConfig: AnyModelConfig) => boolean;
  onSelect: (modelConfig: AnyModelConfig) => void;
  onClose?: () => void;
};

type ModelCmdkState =
  | (ModelCmdkOptions & {
      isOpen: true;
    })
  | {
      isOpen: false;
    };

const $modelCmdkState = atom<ModelCmdkState>({ isOpen: false });

const openModelCmdk = (options: ModelCmdkOptions) => {
  $modelCmdkState.set({
    isOpen: true,
    ...options,
  });
};

const closeModelCmdk = () => {
  $modelCmdkState.set({ isOpen: false });
};

const regexCache = new LRUCache<string, RegExp>({ max: 1000 });

const getRegex = (searchTerm: string) => {
  const cachedRegex = regexCache.get(searchTerm);
  if (cachedRegex) {
    return cachedRegex;
  }
  const regex = new RegExp(
    searchTerm
      .trim()
      .replace(/[-[\]{}()*+!<=:?./\\^$|#,]/g, '')
      .split(' ')
      .join('.*'),
    'gi'
  );
  regexCache.set(searchTerm, regex);

  return regex;
};

const filterCache = new LRUCache<string, boolean>({ max: 1000 });
const getFilter = (model: AnyModelConfig, searchTerm: string) => {
  const key = `${model.key}-${searchTerm}`;
  const cachedFilter = filterCache.get(key);
  if (cachedFilter !== undefined) {
    return cachedFilter;
  }

  if (!searchTerm) {
    filterCache.set(key, true);
    return true;
  }

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
    filterCache.set(key, true);
    return true;
  }

  filterCache.set(key, false);
  return false;
};

const useEnrichedModelConfigs = () => {
  const { data } = useGetModelConfigsQuery();
  const models = useMemo(() => {
    if (!data || data.ids.length === 0) {
      return EMPTY_ARRAY;
    }
    const allModels = modelConfigsAdapterSelectors.selectAll(data);
    const enrichedModels: (AnyModelConfig & { searchableContent: string })[] = allModels.map((model) => {
      const searchableContent = [model.name, model.base, model.type, model.format, model.description ?? ''].join(' ');
      return {
        ...model,
        searchableContent,
      };
    });
    return enrichedModels;
  }, [data]);
  return models;
};

export const useModelCmdk = (options: ModelCmdkOptions) => {
  const onOpen = useCallback(() => {
    openModelCmdk(options);
  }, [options]);
  const onClose = useCallback(() => {
    closeModelCmdk();
  }, []);
  return {
    onOpen,
    onClose,
  };
};

const cmdkRootSx: SystemStyleObject = {
  p: 2,
  h: 'full',
  '[cmdk-root]': {
    w: 'full',
    h: 'full',
  },
  '[cmdk-list]': {
    w: 'full',
    h: 'full',
  },
};

export const ModelCmdk = memo(() => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const state = useStore($modelCmdkState);
  // Filtering the list is expensive - debounce the search term to avoid stutters
  const [debouncedSearchTerm] = useDebounce(searchTerm, 300);

  const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  }, []);

  const onClose = useCallback(() => {
    closeModelCmdk();
    setSearchTerm('');
  }, []);

  const onSelect = useCallback(
    (model: AnyModelConfig) => {
      if (!state.isOpen) {
        // If the command menu is closed, we shouldn't do anything
        return;
      }
      state.onSelect(model);
      onClose();
    },
    [onClose, state]
  );

  return (
    <Modal isOpen={state.isOpen} onClose={onClose} useInert={false} initialFocusRef={inputRef} size="xl" isCentered>
      <ModalOverlay />
      <ModalContent h="512" maxH="70%">
        <ModalBody sx={cmdkRootSx}>
          {state.isOpen && (
            <CommandRoot loop shouldFilter={false}>
              <Flex flexDir="column" h="full" gap={2}>
                <Input ref={inputRef} value={searchTerm} onChange={onChange} placeholder={t('nodes.nodeSearch')} />
                <Box w="full" h="full">
                  <ScrollableContent>
                    <CommandEmpty>
                      <IAINoContentFallback
                        position="absolute"
                        top={0}
                        right={0}
                        bottom={0}
                        left={0}
                        icon={null}
                        label="No matching items"
                      />
                    </CommandEmpty>
                    <CommandList>
                      <ModelList searchTerm={debouncedSearchTerm} onSelect={onSelect} filter={state.filter} />
                    </CommandList>
                  </ScrollableContent>
                </Box>
              </Flex>
            </CommandRoot>
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

ModelCmdk.displayName = 'ModelCmdk';

const ModelList = memo(
  (props: {
    searchTerm: string;
    filter?: (model: AnyModelConfig) => boolean;
    onSelect: (model: AnyModelConfig) => void;
  }) => {
    const { data } = useGetModelConfigsQuery();
    const filteredModels = useMemo(() => {
      if (!data || data.ids.length === 0) {
        return EMPTY_ARRAY;
      }
      const allModels = modelConfigsAdapterSelectors.selectAll(data);
      return props.filter ? allModels.filter(props.filter) : allModels;
    }, [data, props.filter]);
    const results = useMemo(() => {
      if (!props.searchTerm) {
        return filteredModels;
      }
      const results: AnyModelConfig[] = [];
      for (const model of filteredModels) {
        if (getFilter(model, props.searchTerm)) {
          results.push(model);
        }
      }
      return results;
    }, [filteredModels, props.searchTerm]);
    const onSelect = useCallback(
      (key: string) => {
        if (!data) {
          // Data not loaded? No models? We should never get here.
          return;
        }
        const model = modelConfigsAdapterSelectors.selectById(data, key);
        if (!model) {
          // Model not found? We should never get here.
          return;
        }
        props.onSelect(model);
      },
      [data, props]
    );

    return (
      <>
        {results.map((model) => (
          <ModelItem key={model.key} model={model} onSelect={onSelect} />
        ))}
      </>
    );
  }
);
ModelList.displayName = 'ModelList';

const cmdkItemSx: SystemStyleObject = {
  flexDir: 'column',
  py: 1,
  px: 2,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
  },
};

const ChakraCommandItem = chakra(CommandItem);

const ModelItem = memo((props: { model: AnyModelConfig; onSelect: (key: string) => void }) => {
  const { model, onSelect } = props;
  return (
    <ChakraCommandItem value={model.key} onSelect={onSelect} role="button" sx={cmdkItemSx}>
      <Flex alignItems="center" gap={2}>
        <Text fontWeight="semibold">{model.name}</Text>
        <Spacer />
        <Text variant="subtext" fontWeight="semibold">
          {model.base}
        </Text>
      </Flex>
      {model.description && <Text color="base.200">{model.description}</Text>}
    </ChakraCommandItem>
  );
});
ModelItem.displayName = 'ModelItem';
