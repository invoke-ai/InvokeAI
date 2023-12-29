import { Flex } from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { difference, forEach, intersection, map, values } from 'lodash-es';
import type { ChangeEvent, MouseEvent } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import type { SearchFolderResponse } from 'services/api/endpoints/models';
import {
  useGetMainModelsQuery,
  useGetModelsInFolderQuery,
  useImportMainModelsMutation,
} from 'services/api/endpoints/models';

export default function FoundModelsList() {
  const searchFolder = useAppSelector(
    (state: RootState) => state.modelmanager.searchFolder
  );
  const [nameFilter, setNameFilter] = useState<string>('');

  // Get paths of models that are already installed
  const { data: installedModels } = useGetMainModelsQuery(ALL_BASE_MODELS);

  // Get all model paths from a given directory
  const { foundModels, alreadyInstalled, filteredModels } =
    useGetModelsInFolderQuery(
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

  const handleClickSetAdvanced = useCallback(
    (model: string) => dispatch(setAdvancedAddScanModel(model)),
    [dispatch]
  );

  const renderModels = ({
    models,
    showActions = true,
  }: {
    models: string[];
    showActions?: boolean;
  }) => {
    return models.map((model) => {
      return (
        <Flex
          key={model}
          p={4}
          gap={4}
          alignItems="center"
          borderRadius={4}
          bg="base.800"
        >
          <Flex w="full" minW="25%" flexDir="column">
            <InvText fontWeight="semibold">
              {model.split('\\').slice(-1)[0]}
            </InvText>
            <InvText fontSize="sm" color="base.400">
              {model}
            </InvText>
          </Flex>
          {showActions ? (
            <Flex gap={2}>
              <InvButton
                id={model}
                onClick={quickAddHandler}
                isLoading={isLoading}
              >
                {t('modelManager.quickAdd')}
              </InvButton>
              <InvButton
                onClick={handleClickSetAdvanced.bind(null, model)}
                isLoading={isLoading}
              >
                {t('modelManager.advanced')}
              </InvButton>
            </Flex>
          ) : (
            <InvText
              fontWeight="semibold"
              p={2}
              borderRadius={4}
              color="blue.100"
              bg="blue.600"
            >
              {t('common.installed')}
            </InvText>
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
        <Flex
          w="full"
          h="full"
          justifyContent="center"
          alignItems="center"
          height={96}
          userSelect="none"
          bg="base.900"
        >
          <InvText variant="subtext">{t('modelManager.noModels')}</InvText>
        </Flex>
      );
    }

    return (
      <Flex flexDirection="column" gap={2} w="100%" minW="50%">
        <InvControl label={t('modelManager.search')}>
          <InvInput onChange={handleSearchFilter} />
        </InvControl>
        <Flex p={2} gap={2}>
          <InvText fontWeight="semibold">
            {t('modelManager.modelsFound')}: {foundModels.length}
          </InvText>
          <InvText fontWeight="semibold" color="blue.200">
            {t('common.notInstalled')}: {filteredModels.length}
          </InvText>
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
}

const foundModelsFilter = (
  data: SearchFolderResponse | undefined,
  nameFilter: string
) => {
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
