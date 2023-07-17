import { Flex, Text } from '@chakra-ui/react';
import { makeToast } from 'app/components/Toaster';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { addToast } from 'features/system/store/systemSlice';
import { difference, forEach, map, values } from 'lodash-es';
import { ChangeEvent, MouseEvent, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  SearchFolderResponse,
  useGetMainModelsQuery,
  useGetModelsInFolderQuery,
  useImportMainModelsMutation,
} from 'services/api/endpoints/models';
import { setAdvancedAddScanModel } from '../../store/modelManagerSlice';

export default function FoundModelsList() {
  const searchFolder = useAppSelector(
    (state: RootState) => state.modelmanager.searchFolder
  );
  const [nameFilter, setNameFilter] = useState<string>('');

  // Get paths of models that are already installed
  const { data: installedModels } = useGetMainModelsQuery();

  // Get all model paths from a given directory
  const { foundModels, notInstalledModels, filteredModels } =
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
          return {
            foundModels: data,
            notInstalledModels: notInstalledModels,
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
                  title: 'Faile To Add Model',
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

  const renderFoundModels = () => {
    if (!searchFolder) return;

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
        />
        <Flex p={2} gap={2}>
          <Text
            sx={{
              fontWeight: 600,
              color: 'accent.500',
              _dark: {
                color: 'accent.200',
              },
            }}
          >
            Found Models: {foundModels.length}
          </Text>
          <Text sx={{ fontWeight: 600 }}>
            Not Installed: {notInstalledModels.length}
          </Text>
        </Flex>

        {filteredModels.map((model) => (
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
          </Flex>
        ))}
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
      return;
    }

    if (model.includes(nameFilter)) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};
