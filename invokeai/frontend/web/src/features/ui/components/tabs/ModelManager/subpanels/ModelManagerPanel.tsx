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

    const openedModelData = mainModels['entities'][openModel];

    if (openedModelData && openedModelData.model_format === 'diffusers') {
      return (
        <DiffusersModelEdit
          modelToEdit={openModel}
          retrievedModel={openedModelData}
          key={openModel}
        />
      );
    }

    if (openedModelData && openedModelData.model_format === 'checkpoint') {
      return (
        <CheckpointModelEdit
          modelToEdit={openModel}
          retrievedModel={openedModelData}
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
