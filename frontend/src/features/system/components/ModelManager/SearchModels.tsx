import React from 'react';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';

import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import { Box, Flex, FormControl, HStack, Text, VStack } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { useTranslation } from 'react-i18next';

import { FaPlus } from 'react-icons/fa';
import { MdFindInPage } from 'react-icons/md';

import { addNewModel, searchForModels } from 'app/socketio/actions';
import {
  setFoundModels,
  setSearchFolder,
} from 'features/system/store/systemSlice';
import { setShouldShowExistingModelsInSearch } from 'features/options/store/optionsSlice';

import _ from 'lodash';

import type { RootState } from 'app/store';
import type { ReactNode, ChangeEvent } from 'react';
import type { FoundModel } from 'app/invokeai';
import IAIInput from 'common/components/IAIInput';
import { Field, Formik } from 'formik';
import { BiReset } from 'react-icons/bi';

const existingModelsSelector = createSelector([systemSelector], (system) => {
  const { model_list } = system;

  const existingModels: string[] = [];

  _.forEach(model_list, (value) => {
    existingModels.push(value.weights);
  });

  return existingModels;
});

function ModelExistsTag() {
  const { t } = useTranslation();
  return (
    <Box
      position={'absolute'}
      zIndex={2}
      right={4}
      top={4}
      fontSize="0.7rem"
      fontWeight={'bold'}
      backgroundColor={'var(--accent-color)'}
      padding={'0.2rem 0.5rem'}
      borderRadius="0.2rem"
      alignItems={'center'}
    >
      {t('modelmanager:modelExists')}
    </Box>
  );
}

interface SearchModelEntry {
  model: FoundModel;
  modelsToAdd: string[];
  setModelsToAdd: React.Dispatch<React.SetStateAction<string[]>>;
}

function SearchModelEntry({
  model,
  modelsToAdd,
  setModelsToAdd,
}: SearchModelEntry) {
  const existingModels = useAppSelector(existingModelsSelector);

  const foundModelsChangeHandler = (e: ChangeEvent<HTMLInputElement>) => {
    if (!modelsToAdd.includes(e.target.value)) {
      setModelsToAdd([...modelsToAdd, e.target.value]);
    } else {
      setModelsToAdd(_.remove(modelsToAdd, (v) => v !== e.target.value));
    }
  };

  return (
    <Box position="relative">
      {existingModels.includes(model.location) ? <ModelExistsTag /> : null}
      <IAICheckbox
        value={model.name}
        label={
          <>
            <VStack alignItems={'start'}>
              <p style={{ fontWeight: 'bold' }}>{model.name}</p>
              <p style={{ fontStyle: 'italic' }}>{model.location}</p>
            </VStack>
          </>
        }
        isChecked={modelsToAdd.includes(model.name)}
        isDisabled={existingModels.includes(model.location)}
        onChange={foundModelsChangeHandler}
        padding={'1rem'}
        backgroundColor={'var(--background-color)'}
        borderRadius={'0.5rem'}
        _checked={{
          backgroundColor: 'var(--accent-color)',
          color: 'var(--text-color)',
        }}
        _disabled={{
          backgroundColor: 'var(--background-color-secondary)',
        }}
      ></IAICheckbox>
    </Box>
  );
}

export default function SearchModels() {
  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const searchFolder = useAppSelector(
    (state: RootState) => state.system.searchFolder
  );

  const foundModels = useAppSelector(
    (state: RootState) => state.system.foundModels
  );

  const existingModels = useAppSelector(existingModelsSelector);

  const shouldShowExistingModelsInSearch = useAppSelector(
    (state: RootState) => state.options.shouldShowExistingModelsInSearch
  );

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const [modelsToAdd, setModelsToAdd] = React.useState<string[]>([]);

  const resetSearchModelHandler = () => {
    dispatch(setSearchFolder(null));
    dispatch(setFoundModels(null));
    setModelsToAdd([]);
  };

  const findModelsHandler = (values: { checkpointFolder: string }) => {
    dispatch(searchForModels(values.checkpointFolder));
  };

  const addAllToSelected = () => {
    setModelsToAdd([]);
    if (foundModels) {
      foundModels.forEach((model) => {
        if (!existingModels.includes(model.location)) {
          setModelsToAdd((currentModels) => {
            return [...currentModels, model.name];
          });
        }
      });
    }
  };

  const removeAllFromSelected = () => {
    setModelsToAdd([]);
  };

  const addSelectedModels = () => {
    const modelsToBeAdded = foundModels?.filter((foundModel) =>
      modelsToAdd.includes(foundModel.name)
    );
    modelsToBeAdded?.forEach((model) => {
      const modelFormat = {
        name: model.name,
        description: '',
        config: 'configs/stable-diffusion/v1-inference.yaml',
        weights: model.location,
        vae: '',
        width: 512,
        height: 512,
        default: false,
      };
      dispatch(addNewModel(modelFormat));
    });
    setModelsToAdd([]);
  };

  const renderFoundModels = () => {
    const newFoundModels: ReactNode[] = [];
    const existingFoundModels: ReactNode[] = [];

    if (foundModels) {
      foundModels.forEach((model, index) => {
        if (existingModels.includes(model.location)) {
          existingFoundModels.push(
            <SearchModelEntry
              key={index}
              model={model}
              modelsToAdd={modelsToAdd}
              setModelsToAdd={setModelsToAdd}
            />
          );
        } else {
          newFoundModels.push(
            <SearchModelEntry
              key={index}
              model={model}
              modelsToAdd={modelsToAdd}
              setModelsToAdd={setModelsToAdd}
            />
          );
        }
      });
    }

    return (
      <>
        {newFoundModels}
        {shouldShowExistingModelsInSearch && existingFoundModels}
      </>
    );
  };

  return (
    <>
      {searchFolder ? (
        <Flex
          flexDirection={'column'}
          padding={'1rem'}
          backgroundColor={'var(--background-color)'}
          borderRadius="0.5rem"
          rowGap={'0.5rem'}
          position={'relative'}
        >
          <p
            style={{
              fontWeight: 'bold',
              fontSize: '0.8rem',
              backgroundColor: 'var(--background-color-secondary)',
              padding: '0.2rem 1rem',
              width: 'max-content',
              borderRadius: '0.2rem',
            }}
          >
            {t('modelmanager:checkpointFolder')}
          </p>
          <p
            style={{ fontWeight: 'bold', fontSize: '0.8rem', maxWidth: '80%' }}
          >
            {searchFolder}
          </p>
          <IAIIconButton
            aria-label={t('modelmanager:scanAgain')}
            tooltip={t('modelmanager:scanAgain')}
            icon={<BiReset />}
            position={'absolute'}
            right={16}
            fontSize={18}
            disabled={isProcessing}
            onClick={() => dispatch(searchForModels(searchFolder))}
          />
          <IAIIconButton
            aria-label={t('modelmanager:clearCheckpointFolder')}
            icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
            position={'absolute'}
            right={5}
            onClick={resetSearchModelHandler}
          />
        </Flex>
      ) : (
        <Formik
          initialValues={{ checkpointFolder: '' }}
          onSubmit={(values) => {
            findModelsHandler(values);
          }}
        >
          {({ handleSubmit }) => (
            <form onSubmit={handleSubmit}>
              <HStack columnGap="0.5rem">
                <FormControl isRequired width="max-content">
                  <Field
                    as={IAIInput}
                    id="checkpointFolder"
                    name="checkpointFolder"
                    type="text"
                    width="lg"
                    size="md"
                    label={t('modelmanager:checkpointFolder')}
                  />
                </FormControl>
                <IAIIconButton
                  icon={<MdFindInPage />}
                  aria-label={t('modelmanager:findModels')}
                  tooltip={t('modelmanager:findModels')}
                  type="submit"
                  disabled={isProcessing}
                />
              </HStack>
            </form>
          )}
        </Formik>
      )}
      {foundModels && (
        <Flex flexDirection={'column'} rowGap={'1rem'}>
          <Flex justifyContent={'space-between'} alignItems="center">
            <p>
              {t('modelmanager:modelsFound')}: {foundModels.length}
            </p>
            <p>
              {t('modelmanager:selected')}: {modelsToAdd.length}
            </p>
          </Flex>
          <Flex columnGap={'0.5rem'} justifyContent={'space-between'}>
            <Flex columnGap={'0.5rem'}>
              <IAIButton
                isDisabled={modelsToAdd.length === foundModels.length}
                onClick={addAllToSelected}
              >
                {t('modelmanager:selectAll')}
              </IAIButton>
              <IAIButton
                isDisabled={modelsToAdd.length === 0}
                onClick={removeAllFromSelected}
              >
                {t('modelmanager:deselectAll')}
              </IAIButton>
              <IAICheckbox
                label={t('modelmanager:showExisting')}
                isChecked={shouldShowExistingModelsInSearch}
                onChange={() =>
                  dispatch(
                    setShouldShowExistingModelsInSearch(
                      !shouldShowExistingModelsInSearch
                    )
                  )
                }
              />
            </Flex>

            <IAIButton
              isDisabled={modelsToAdd.length === 0}
              onClick={addSelectedModels}
              backgroundColor={
                modelsToAdd.length > 0 ? 'var(--accent-color) !important' : ''
              }
            >
              {t('modelmanager:addSelected')}
            </IAIButton>
          </Flex>
          <Flex
            rowGap="1rem"
            flexDirection="column"
            maxHeight="18rem"
            overflowY="scroll"
            paddingRight="1rem"
            paddingLeft="0.2rem"
            borderRadius="0.2rem"
          >
            {foundModels.length > 0 ? (
              modelsToAdd.length === 0 && (
                <Text
                  fontWeight="bold"
                  fontSize={14}
                  padding="0.5rem"
                  borderRadius="0.2rem"
                  margin="0 0.5rem 0 1rem"
                  textAlign="center"
                  backgroundColor="var(--notice-color)"
                  boxShadow="0 0 200px 6px var(--notice-color)"
                  marginTop="1rem"
                  width="max-content"
                >
                  {t('modelmanager:selectAndAdd')}
                </Text>
              )
            ) : (
              <Text
                fontWeight="bold"
                fontSize={14}
                padding="0.5rem"
                borderRadius="0.2rem"
                textAlign="center"
                backgroundColor="var(--status-bad-color)"
              >
                {t('modelmanager:noModelsFound')}
              </Text>
            )}

            {renderFoundModels()}
          </Flex>
        </Flex>
      )}
    </>
  );
}
