import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIIconButton from 'common/components/IAIIconButton';
import React from 'react';

import {
  Badge,
  Flex,
  FormControl,
  HStack,
  Radio,
  RadioGroup,
  Spacer,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { useTranslation } from 'react-i18next';

import { FaSearch, FaTrash } from 'react-icons/fa';

// import { addNewModel, searchForModels } from 'app/socketio/actions';
import {
  setFoundModels,
  setSearchFolder,
} from 'features/system/store/systemSlice';
import { setShouldShowExistingModelsInSearch } from 'features/ui/store/uiSlice';

import type { FoundModel } from 'app/types/invokeai';
import type { RootState } from 'app/store/store';
import IAIInput from 'common/components/IAIInput';
import { Field, Formik } from 'formik';
import { forEach, remove } from 'lodash-es';
import type { ChangeEvent, ReactNode } from 'react';
import IAIForm from 'common/components/IAIForm';

const existingModelsSelector = createSelector([systemSelector], (system) => {
  const { model_list } = system;

  const existingModels: string[] = [];

  forEach(model_list, (value) => {
    existingModels.push(value.weights);
  });

  return existingModels;
});

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
  const { t } = useTranslation();
  const existingModels = useAppSelector(existingModelsSelector);

  const foundModelsChangeHandler = (e: ChangeEvent<HTMLInputElement>) => {
    if (!modelsToAdd.includes(e.target.value)) {
      setModelsToAdd([...modelsToAdd, e.target.value]);
    } else {
      setModelsToAdd(remove(modelsToAdd, (v) => v !== e.target.value));
    }
  };

  return (
    <Flex
      flexDirection="column"
      gap={2}
      backgroundColor={
        modelsToAdd.includes(model.name) ? 'accent.650' : 'base.800'
      }
      paddingX={4}
      paddingY={2}
      borderRadius={4}
    >
      <Flex gap={4} alignItems="center" justifyContent="space-between">
        <IAICheckbox
          value={model.name}
          label={<Text fontWeight={500}>{model.name}</Text>}
          isChecked={modelsToAdd.includes(model.name)}
          isDisabled={existingModels.includes(model.location)}
          onChange={foundModelsChangeHandler}
        ></IAICheckbox>
        {existingModels.includes(model.location) && (
          <Badge colorScheme="accent">{t('modelManager.modelExists')}</Badge>
        )}
      </Flex>
      <Text fontStyle="italic" variant="subtext">
        {model.location}
      </Text>
    </Flex>
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
    (state: RootState) => state.ui.shouldShowExistingModelsInSearch
  );

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const [modelsToAdd, setModelsToAdd] = React.useState<string[]>([]);
  const [modelType, setModelType] = React.useState<string>('v1');
  const [pathToConfig, setPathToConfig] = React.useState<string>('');

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

    const configFiles = {
      v1: 'configs/stable-diffusion/v1-inference.yaml',
      v2_base: 'configs/stable-diffusion/v2-inference-v.yaml',
      v2_768: 'configs/stable-diffusion/v2-inference-v.yaml',
      inpainting: 'configs/stable-diffusion/v1-inpainting-inference.yaml',
      custom: pathToConfig,
    };

    modelsToBeAdded?.forEach((model) => {
      const modelFormat = {
        name: model.name,
        description: '',
        config: configFiles[modelType as keyof typeof configFiles],
        weights: model.location,
        vae: '',
        width: 512,
        height: 512,
        default: false,
        format: 'ckpt',
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
      <Flex flexDirection="column" rowGap={4}>
        {newFoundModels}
        {shouldShowExistingModelsInSearch && existingFoundModels}
      </Flex>
    );
  };

  return (
    <>
      {searchFolder ? (
        <Flex
          sx={{
            padding: 4,
            gap: 2,
            position: 'relative',
            borderRadius: 'base',
            alignItems: 'center',
            w: 'full',
            bg: 'base.900',
          }}
        >
          <Flex
            sx={{
              flexDir: 'column',
              gap: 2,
            }}
          >
            <Text
              sx={{
                fontWeight: 500,
              }}
              variant="subtext"
            >
              {t('modelManager.checkpointFolder')}
            </Text>
            <Text sx={{ fontWeight: 500 }}>{searchFolder}</Text>
          </Flex>
          <Spacer />
          <IAIIconButton
            aria-label={t('modelManager.scanAgain')}
            tooltip={t('modelManager.scanAgain')}
            icon={<FaSearch />}
            fontSize={18}
            disabled={isProcessing}
            onClick={() => dispatch(searchForModels(searchFolder))}
          />
          <IAIIconButton
            aria-label={t('modelManager.clearCheckpointFolder')}
            tooltip={t('modelManager.clearCheckpointFolder')}
            icon={<FaTrash />}
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
            <IAIForm onSubmit={handleSubmit} width="100%">
              <HStack columnGap={2} alignItems="flex-end">
                <FormControl flexGrow={1}>
                  <Field
                    as={IAIInput}
                    id="checkpointFolder"
                    name="checkpointFolder"
                    type="text"
                    size="md"
                    label={t('modelManager.checkpointFolder')}
                  />
                </FormControl>
                <IAIButton
                  leftIcon={<FaSearch />}
                  aria-label={t('modelManager.findModels')}
                  tooltip={t('modelManager.findModels')}
                  type="submit"
                  disabled={isProcessing}
                  px={8}
                >
                  {t('modelManager.findModels')}
                </IAIButton>
              </HStack>
            </IAIForm>
          )}
        </Formik>
      )}
      {foundModels && (
        <Flex flexDirection="column" rowGap={4} width="full">
          <Flex justifyContent="space-between" alignItems="center">
            <p>
              {t('modelManager.modelsFound')}: {foundModels.length}
            </p>
            <p>
              {t('modelManager.selected')}: {modelsToAdd.length}
            </p>
          </Flex>
          <Flex columnGap={2} justifyContent="space-between">
            <Flex columnGap={2}>
              <IAIButton
                isDisabled={modelsToAdd.length === foundModels.length}
                onClick={addAllToSelected}
              >
                {t('modelManager.selectAll')}
              </IAIButton>
              <IAIButton
                isDisabled={modelsToAdd.length === 0}
                onClick={removeAllFromSelected}
              >
                {t('modelManager.deselectAll')}
              </IAIButton>
              <IAICheckbox
                label={t('modelManager.showExisting')}
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
              colorScheme="accent"
            >
              {t('modelManager.addSelected')}
            </IAIButton>
          </Flex>

          <Flex
            sx={{
              flexDirection: 'column',
              padding: 4,
              rowGap: 4,
              borderRadius: 'base',
              width: 'full',
              bg: 'base.900',
            }}
          >
            <Flex gap={4}>
              <Text fontWeight={500} variant="subtext">
                {t('modelManager.pickModelType')}
              </Text>
              <RadioGroup
                value={modelType}
                onChange={(v) => setModelType(v)}
                defaultValue="v1"
                name="model_type"
              >
                <Flex gap={4}>
                  <Radio value="v1">
                    <Text fontSize="sm">{t('modelManager.v1')}</Text>
                  </Radio>
                  <Radio value="v2_base">
                    <Text fontSize="sm">{t('modelManager.v2_base')}</Text>
                  </Radio>
                  <Radio value="v2_768">
                    <Text fontSize="sm">{t('modelManager.v2_768')}</Text>
                  </Radio>
                  <Radio value="inpainting">
                    <Text fontSize="sm">{t('modelManager.inpainting')}</Text>
                  </Radio>
                  <Radio value="custom">
                    <Text fontSize="sm">{t('modelManager.customConfig')}</Text>
                  </Radio>
                </Flex>
              </RadioGroup>
            </Flex>

            {modelType === 'custom' && (
              <Flex flexDirection="column" rowGap={2}>
                <Text fontWeight="500" fontSize="sm" variant="subtext">
                  {t('modelManager.pathToCustomConfig')}
                </Text>
                <IAIInput
                  value={pathToConfig}
                  onChange={(e) => {
                    if (e.target.value !== '') setPathToConfig(e.target.value);
                  }}
                  width="full"
                />
              </Flex>
            )}
          </Flex>

          <Flex
            flexDirection="column"
            maxHeight={72}
            overflowY="scroll"
            borderRadius="sm"
            gap={2}
          >
            {foundModels.length > 0 ? (
              renderFoundModels()
            ) : (
              <Text
                fontWeight="500"
                padding={2}
                borderRadius="sm"
                textAlign="center"
                variant="subtext"
              >
                {t('modelManager.noModelsFound')}
              </Text>
            )}
          </Flex>
        </Flex>
      )}
    </>
  );
}
