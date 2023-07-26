import { Flex, Text } from '@chakra-ui/react';
import { makeToast } from 'features/system/util/makeToast';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAIScrollArea from 'common/components/IAIScrollArea';
import { addToast } from 'features/system/store/systemSlice';
import { difference, forEach, intersection, map, values } from 'lodash-es';
import { ChangeEvent, MouseEvent, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  SearchFolderResponse,
  useGetMainModelsQuery,
  useGetModelsInFolderQuery,
  useImportMainModelsMutation,
} from 'services/api/endpoints/models';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';
import { ALL_BASE_MODELS } from 'services/api/constants';

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
                  title: 'Failed To Add Model',
                  status: 'error',
                })
              )
            );
          }
        });
    },
    [dispatch, importMainModel]
  );

  const handleSearchFilter = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNameFilter(e.target.value);
  }, []);

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
          sx={{
            p: 4,
            gap: 4,
            alignItems: 'center',
            borderRadius: 4,
            bg: 'base.200',
            _dark: {
              bg: 'base.800',
            },
          }}
          key={model}
        >
          <Flex w="100%" sx={{ flexDirection: 'column', minW: '25%' }}>
            <Text sx={{ fontWeight: 600 }}>
              {model.split('\\').slice(-1)[0]}
            </Text>
            <Text
              sx={{
                fontSize: 'sm',
                color: 'base.600',
                _dark: {
                  color: 'base.400',
                },
              }}
            >
              {model}
            </Text>
          </Flex>
          {showActions ? (
            <Flex gap={2}>
              <IAIButton
                id={model}
                onClick={quickAddHandler}
                isLoading={isLoading}
              >
                Quick Add
              </IAIButton>
              <IAIButton
                onClick={() => dispatch(setAdvancedAddScanModel(model))}
                isLoading={isLoading}
              >
                Advanced
              </IAIButton>
            </Flex>
          ) : (
            <Text
              sx={{
                fontWeight: 600,
                p: 2,
                borderRadius: 4,
                color: 'accent.50',
                bg: 'accent.400',
                _dark: {
                  color: 'accent.100',
                  bg: 'accent.600',
                },
              }}
            >
              Installed
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
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            justifyContent: 'center',
            alignItems: 'center',
            height: 96,
            userSelect: 'none',
            bg: 'base.200',
            _dark: {
              bg: 'base.900',
            },
          }}
        >
          <Text variant="subtext">No Models Found</Text>
        </Flex>
      );
    }

    return (
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 2,
          w: '100%',
          minW: '50%',
        }}
      >
        <IAIInput
          onChange={handleSearchFilter}
          label={t('modelManager.search')}
          labelPos="side"
        />
        <Flex p={2} gap={2}>
          <Text sx={{ fontWeight: 600 }}>
            Models Found: {foundModels.length}
          </Text>
          <Text
            sx={{
              fontWeight: 600,
              color: 'accent.500',
              _dark: {
                color: 'accent.200',
              },
            }}
          >
            Not Installed: {filteredModels.length}
          </Text>
        </Flex>

        <IAIScrollArea offsetScrollbars>
          <Flex gap={2} flexDirection="column">
            {renderModels({ models: filteredModels })}
            {renderModels({ models: alreadyInstalled, showActions: false })}
          </Flex>
        </IAIScrollArea>
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
