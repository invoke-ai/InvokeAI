import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';

import { useGetMainModelsQuery } from 'services/api/endpoints/models';
import CheckpointModelEdit from './ModelManagerPanel/CheckpointModelEdit';
import DiffusersModelEdit from './ModelManagerPanel/DiffusersModelEdit';
import ModelList from './ModelManagerPanel/ModelList';

export default function ModelManagerPanel() {
  const { data: mainModels } = useGetMainModelsQuery();

  const openModel = useAppSelector(
    (state: RootState) => state.system.openModel
  );

  const renderModelEditTabs = () => {
    if (!openModel || !mainModels) return;

    if (mainModels['entities'][openModel]['model_format'] === 'diffusers') {
      return (
        <DiffusersModelEdit
          modelToEdit={openModel}
          retrievedModel={mainModels['entities'][openModel]}
          key={openModel}
        />
      );
    } else {
      return (
        <CheckpointModelEdit
          modelToEdit={openModel}
          retrievedModel={mainModels['entities'][openModel]}
          key={openModel}
        />
      );
    }
  };
  return (
    <Flex width="100%" columnGap={8}>
      <ModelList />
      {renderModelEditTabs()}
    </Flex>
  );
}
