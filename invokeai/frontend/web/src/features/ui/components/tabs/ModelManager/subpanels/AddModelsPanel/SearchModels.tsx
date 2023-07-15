import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';
import FoundModelsList from './FoundModelsList';
import SearchFolderForm from './SearchFolderForm';

export default function SearchModels() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <Flex flexDirection="column" w="100%">
      <SearchFolderForm />
      <FoundModelsList />
    </Flex>
  );
}
