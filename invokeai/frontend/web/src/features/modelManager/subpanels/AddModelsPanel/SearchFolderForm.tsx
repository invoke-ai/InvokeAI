import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import {
  setAdvancedAddScanModel,
  setSearchFolder,
} from 'features/modelManager/store/modelManagerSlice';
import type { CSSProperties } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSearch, FaSync, FaTrash } from 'react-icons/fa';
import { useGetModelsInFolderQuery } from 'services/api/endpoints/models';

type SearchFolderForm = {
  folder: string;
};

function SearchFolderForm() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const searchFolder = useAppSelector((s) => s.modelmanager.searchFolder);

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

  const scanAgainHandler = useCallback(() => {
    refetchFoundModels();
  }, [refetchFoundModels]);

  const handleClickClearCheckpointFolder = useCallback(() => {
    dispatch(setSearchFolder(null));
    dispatch(setAdvancedAddScanModel(null));
  }, [dispatch]);

  return (
    <form
      onSubmit={searchFolderForm.onSubmit((values) =>
        searchFolderFormSubmitHandler(values)
      )}
      style={formStyles}
    >
      <Flex w="100%" gap={2} borderRadius={4} alignItems="center">
        <Flex w="100%" alignItems="center" gap={4} minH={12}>
          <InvText
            fontSize="sm"
            fontWeight="semibold"
            color="base.300"
            minW="max-content"
          >
            {t('common.folder')}
          </InvText>
          {!searchFolder ? (
            <InvInput
              w="100%"
              size="md"
              {...searchFolderForm.getInputProps('folder')}
            />
          ) : (
            <Flex
              w="100%"
              p={2}
              px={4}
              bg="base.700"
              borderRadius={4}
              fontSize="sm"
              fontWeight="bold"
            >
              {searchFolder}
            </Flex>
          )}
        </Flex>

        <Flex gap={2}>
          {!searchFolder ? (
            <InvIconButton
              aria-label={t('modelManager.findModels')}
              tooltip={t('modelManager.findModels')}
              icon={<FaSearch />}
              fontSize={18}
              size="sm"
              type="submit"
            />
          ) : (
            <InvIconButton
              aria-label={t('modelManager.scanAgain')}
              tooltip={t('modelManager.scanAgain')}
              icon={<FaSync />}
              onClick={scanAgainHandler}
              fontSize={18}
              size="sm"
            />
          )}

          <InvIconButton
            aria-label={t('modelManager.clearCheckpointFolder')}
            tooltip={t('modelManager.clearCheckpointFolder')}
            icon={<FaTrash />}
            size="sm"
            onClick={handleClickClearCheckpointFolder}
            isDisabled={!searchFolder}
            colorScheme="red"
          />
        </Flex>
      </Flex>
    </form>
  );
}

export default memo(SearchFolderForm);

const formStyles: CSSProperties = {
  width: '100%',
};
