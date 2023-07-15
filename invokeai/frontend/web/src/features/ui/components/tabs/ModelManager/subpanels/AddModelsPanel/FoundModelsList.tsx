import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useGetModelsInFolderQuery } from 'services/api/endpoints/models';

export default function FoundModelsList() {
  const searchFolder = useAppSelector(
    (state: RootState) => state.modelmanager.searchFolder
  );

  const { data: foundModels } = useGetModelsInFolderQuery({
    search_path: searchFolder ? searchFolder : '',
  });

  console.log(foundModels);

  const renderFoundModels = () => {
    if (!searchFolder) return;

    if (!foundModels || foundModels.length === 0) {
      return <Flex>No Models Found</Flex>;
    }

    return (
      <Flex
        sx={{
          flexDirection: 'column',
        }}
      >
        {foundModels.map((model) => (
          <Flex key={model}>{model}</Flex>
        ))}
      </Flex>
    );
  };

  return renderFoundModels();
}
