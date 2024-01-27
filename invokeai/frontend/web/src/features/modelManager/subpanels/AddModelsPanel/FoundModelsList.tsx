import { Button, Flex, FormControl, FormLabel, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { difference, forEach, intersection, map, values } from 'lodash-es';
import type { ChangeEvent, MouseEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import type { SearchFolderResponse } from 'services/api/endpoints/models';
import {
  useGetMainModelsQuery,
  useGetModelsInFolderQuery,
  useImportMainModelsMutation,
} from 'services/api/endpoints/models';

const FoundModelsList = () => {
  const searchFolder = useAppSelector((s) => s.modelmanager.searchFolder);
  const [nameFilter, setNameFilter] = useState<string>('');

  // Get paths of models that are already installed
  const { data: installedModels } = useGetMainModelsQuery(ALL_BASE_MODELS);

  // Get all model paths from a given directory
  const { foundModels, alreadyInstalled, filteredModels } = useGetModelsInFolderQuery(
    {
      search_path: searchFolder ? searchFolder : '',
    },
    {
      selectFromResult: ({ data }) => {
        const installedModelValues = values(installedModels?.entities);
        const installedModelPaths = map(installedModelValues, 'path');
        // Only select models those that aren't already installed to Invoke
        const notInstalledModels = difference(data, installedModelPaths);
        const alreadyInstalled = intersection(data, installedModelPaths);
        return {
          foundModels: data,
          alreadyInstalled: foundModelsFilter(alreadyInstalled, nameFilter),
          filteredModels: foundModelsFilter(notInstalledModels, nameFilter),
        };
      },
    }
  );

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const quickAddHandler = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      const model_name = e.currentTarget.id.split('\\').splice(-1)[0];
      importMainModel({
        body: {
          location: e.currentTarget.id,
        },
      })
        .unwrap()
        .then((_) => {
          dispatch(
            addToast(
              makeToast({
                title: `Added Model: ${model_name}`,
                status: 'success',
              })
            )
          );
        })
        .catch((error) => {
          if (error) {
            dispatch(
              addToast(
                makeToast({
                  title: t('toast.modelAddFailed'),
                  status: 'error',
                })
              )
            );
          }
        });
    },
    [dispatch, importMainModel, t]
  );

  const handleSearchFilter = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNameFilter(e.target.value);
  }, []);

  const handleClickSetAdvanced = useCallback((model: string) => dispatch(setAdvancedAddScanModel(model)), [dispatch]);

  const renderModels = ({ models, showActions = true }: { models: string[]; showActions?: boolean }) => {
    return models.map((model) => {
      return (
        <Flex key={model} p={4} gap={4} alignItems="center" borderRadius={4} bg="base.800">
          <Flex w="full" minW="25%" flexDir="column">
            <Text fontWeight="semibold">{model.split('\\').slice(-1)[0]}</Text>
            <Text fontSize="sm" color="base.400">
              {model}
            </Text>
          </Flex>
          {showActions ? (
            <Flex gap={2}>
              <Button id={model} onClick={quickAddHandler} isLoading={isLoading}>
                {t('modelManager.quickAdd')}
              </Button>
              <Button onClick={handleClickSetAdvanced.bind(null, model)} isLoading={isLoading}>
                {t('modelManager.advanced')}
              </Button>
            </Flex>
          ) : (
            <Text fontWeight="semibold" p={2} borderRadius={4} color="invokeBlue.100" bg="invokeBlue.600">
              {t('common.installed')}
            </Text>
          )}
        </Flex>
      );
    });
  };

  const renderFoundModels = () => {
    if (!searchFolder) {
      return null;
    }

    if (!foundModels || foundModels.length === 0) {
      return (
        <Flex w="full" h="full" justifyContent="center" alignItems="center" height={96} userSelect="none" bg="base.900">
          <Text variant="subtext">{t('modelManager.noModels')}</Text>
        </Flex>
      );
    }

    return (
      <Flex flexDirection="column" gap={2} w="100%" minW="50%">
        <FormControl>
          <FormLabel>{t('modelManager.search')}</FormLabel>
          <Input onChange={handleSearchFilter} />
        </FormControl>
        <Flex p={2} gap={2}>
          <Text fontWeight="semibold">
            {t('modelManager.modelsFound')}: {foundModels.length}
          </Text>
          <Text fontWeight="semibold" color="invokeBlue.200">
            {t('common.notInstalled')}: {filteredModels.length}
          </Text>
        </Flex>

        <ScrollableContent>
          <Flex gap={2} flexDirection="column">
            {renderModels({ models: filteredModels })}
            {renderModels({ models: alreadyInstalled, showActions: false })}
          </Flex>
        </ScrollableContent>
      </Flex>
    );
  };

  return renderFoundModels();
};

const foundModelsFilter = (data: SearchFolderResponse | undefined, nameFilter: string) => {
  const filteredModels: SearchFolderResponse = [];
  forEach(data, (model) => {
    if (!model) {
      return null;
    }

    if (model.includes(nameFilter)) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};

export default memo(FoundModelsList);
