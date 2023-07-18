import { Flex, Text } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIInput from 'common/components/IAIInput';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSearch, FaSync, FaTrash } from 'react-icons/fa';
import { useGetModelsInFolderQuery } from 'services/api/endpoints/models';
import {
  setAdvancedAddScanModel,
  setSearchFolder,
} from '../../store/modelManagerSlice';

type SearchFolderForm = {
  folder: string;
};

function SearchFolderForm() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const searchFolder = useAppSelector(
    (state: RootState) => state.modelmanager.searchFolder
  );

  const { refetch: refetchFoundModels } = useGetModelsInFolderQuery({
    search_path: searchFolder ? searchFolder : '',
  });

  const searchFolderForm = useForm<SearchFolderForm>({
    initialValues: {
      folder: '',
    },
  });

  const searchFolderFormSubmitHandler = useCallback(
    (values: SearchFolderForm) => {
      dispatch(setSearchFolder(values.folder));
    },
    [dispatch]
  );

  const scanAgainHandler = () => {
    refetchFoundModels();
  };

  return (
    <form
      onSubmit={searchFolderForm.onSubmit((values) =>
        searchFolderFormSubmitHandler(values)
      )}
      style={{ width: '100%' }}
    >
      <Flex
        sx={{
          w: '100%',
          gap: 2,
          borderRadius: 4,
          alignItems: 'center',
        }}
      >
        <Flex w="100%" alignItems="center" gap={4} minH={12}>
          <Text
            sx={{
              fontSize: 'sm',
              fontWeight: 600,
              color: 'base.700',
              minW: 'max-content',
              _dark: { color: 'base.300' },
            }}
          >
            Folder
          </Text>
          {!searchFolder ? (
            <IAIInput
              w="100%"
              size="md"
              {...searchFolderForm.getInputProps('folder')}
            />
          ) : (
            <Flex
              sx={{
                w: '100%',
                p: 2,
                px: 4,
                bg: 'base.300',
                borderRadius: 4,
                fontSize: 'sm',
                fontWeight: 'bold',
                _dark: { bg: 'base.700' },
              }}
            >
              {searchFolder}
            </Flex>
          )}
        </Flex>

        <Flex gap={2}>
          {!searchFolder ? (
            <IAIIconButton
              aria-label={t('modelManager.findModels')}
              tooltip={t('modelManager.findModels')}
              icon={<FaSearch />}
              fontSize={18}
              size="sm"
              type="submit"
            />
          ) : (
            <IAIIconButton
              aria-label={t('modelManager.scanAgain')}
              tooltip={t('modelManager.scanAgain')}
              icon={<FaSync />}
              onClick={scanAgainHandler}
              fontSize={18}
              size="sm"
            />
          )}

          <IAIIconButton
            aria-label={t('modelManager.clearCheckpointFolder')}
            tooltip={t('modelManager.clearCheckpointFolder')}
            icon={<FaTrash />}
            size="sm"
            onClick={() => {
              dispatch(setSearchFolder(null));
              dispatch(setAdvancedAddScanModel(null));
            }}
            isDisabled={!searchFolder}
            colorScheme="red"
          />
        </Flex>
      </Flex>
    </form>
  );
}

export default memo(SearchFolderForm);
