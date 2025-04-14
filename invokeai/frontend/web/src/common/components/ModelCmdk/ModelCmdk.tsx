import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, chakra, Flex, Input, Modal, ModalBody, ModalContent, ModalOverlay, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { CommandEmpty, CommandItem, CommandList, CommandRoot } from 'cmdk';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { atom } from 'nanostores';
import type { ChangeEvent, RefObject } from 'react';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';
import { useDebounce } from 'use-debounce';

export type ModelCmdkOptions = {
  modelConfigs: AnyModelConfig[];
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
  const inputRef = useRef<HTMLInputElement>(null);
  const state = useStore($modelCmdkState);

  const onSelect = useCallback(
    (model: AnyModelConfig) => {
      if (!state.isOpen) {
        // If the command menu is closed, we shouldn't do anything
        return;
      }
      state.onSelect(model);
      closeModelCmdk();
    },
    [state]
  );

  return (
    <Modal
      isOpen={state.isOpen}
      onClose={closeModelCmdk}
      useInert={false}
      initialFocusRef={inputRef}
      size="xl"
      isCentered
    >
      <ModalOverlay />
      <ModalContent h="512" maxH="70%">
        <ModalBody sx={cmdkRootSx}>
          {state.isOpen && (
            <ModelCommandRoot inputRef={inputRef} modelConfigs={state.modelConfigs} onSelect={onSelect} />
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

ModelCmdk.displayName = 'ModelCmdk';

const ModelCommandRoot = memo(
  (props: {
    inputRef: RefObject<HTMLInputElement>;
    modelConfigs: AnyModelConfig[];
    onSelect: (model: AnyModelConfig) => void;
  }) => {
    const { t } = useTranslation();
    const [value, setValue] = useState('');
    const { inputRef, modelConfigs, onSelect } = props;
    const [searchTerm, setSearchTerm] = useState('');
    // Filtering the list is expensive - debounce the search term to avoid stutters
    const [debouncedSearchTerm] = useDebounce(searchTerm, 300);

    const onChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    }, []);

    return (
      <CommandRoot loop shouldFilter={false} value={value} onValueChange={setValue}>
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
                <ModelList
                  searchTerm={debouncedSearchTerm}
                  onSelect={onSelect}
                  modelConfigs={modelConfigs}
                  setValue={setValue}
                />
              </CommandList>
            </ScrollableContent>
          </Box>
        </Flex>
      </CommandRoot>
    );
  }
);
ModelCommandRoot.displayName = 'ModelCommandRoot';

const ModelList = memo(
  (props: {
    searchTerm: string;
    modelConfigs: AnyModelConfig[];
    onSelect: (model: AnyModelConfig) => void;
    setValue: (value: string) => void;
  }) => {
    const { searchTerm, modelConfigs, onSelect: _onSelect, setValue } = props;

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
    const results = useMemo(() => {
      if (!searchTerm) {
        setValue(modelConfigs[0]?.key ?? '');
        return modelConfigs;
      }
      const results = modelConfigs.filter((model) => isMatch(model, searchTerm));
      setValue(results[0]?.key ?? '');
      return results;
    }, [modelConfigs, searchTerm, setValue]);

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
  display: 'flex',
  flexDir: 'column',
  py: 1,
  px: 2,
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
  },
  '.model-header': {
    display: 'flex',
    gap: 2,
    alignItems: 'center',
    justifyContent: 'space-between',
    'model-name': {
      fontSize: 'sm',
    },
    'model-base': {
      fontSize: 'sm',
      color: 'base.500',
    },
    'model-desc': {
      color: 'base.200',
    },
  },
};

const ChakraCommandItem = chakra(CommandItem);

const ModelItem = memo((props: { model: AnyModelConfig; onSelect: (key: string) => void }) => {
  const { model, onSelect } = props;
  return (
    <ChakraCommandItem value={model.key} onSelect={onSelect} role="button" sx={cmdkItemSx}>
      <Box className="model-header">
        <Text className="model-name">{model.name}</Text>
        <Text className="model-base">{model.base}</Text>
      </Box>
      {model.description && <Text className="model-desc">{model.description}</Text>}
    </ChakraCommandItem>
  );
});
ModelItem.displayName = 'ModelItem';
