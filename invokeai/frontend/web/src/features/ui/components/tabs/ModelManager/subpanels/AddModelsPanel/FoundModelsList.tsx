import { Flex, Text } from '@chakra-ui/react';
import { makeToast } from 'app/components/Toaster';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import { addToast } from 'features/system/store/systemSlice';
import { difference, map, values } from 'lodash-es';
import { MouseEvent, useCallback } from 'react';
import {
  useGetMainModelsQuery,
  useGetModelsInFolderQuery,
  useImportMainModelsMutation,
} from 'services/api/endpoints/models';

export default function FoundModelsList() {
  const searchFolder = useAppSelector(
    (state: RootState) => state.modelmanager.searchFolder
  );

  // Get all model paths from a given directory
  const { data: foundModels } = useGetModelsInFolderQuery({
    search_path: searchFolder ? searchFolder : '',
  });

  // Get paths of models that are already installed
  const { data: installedModels } = useGetMainModelsQuery();
  const installedModelValues = values(installedModels?.entities);
  const installedModelPaths = map(installedModelValues, 'path');

  // Only select models those that aren't already installed to Invoke
  const notInstalledModels = difference(foundModels, installedModelPaths);

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();
  const dispatch = useAppDispatch();

  const quickAddHandler = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
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
                title: 'Added Model',
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

  const renderFoundModels = () => {
    if (!searchFolder) return;

    if (!foundModels || foundModels.length === 0) {
      return <Flex>No Models Found</Flex>;
    }

    return (
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 2,
          w: '100%',
        }}
      >
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

        {notInstalledModels.map((model) => (
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
            <Flex w="100%" sx={{ flexDirection: 'column' }}>
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
              <IAIButton>Advanced</IAIButton>
            </Flex>
          </Flex>
        ))}
      </Flex>
    );
  };

  return renderFoundModels();
}
