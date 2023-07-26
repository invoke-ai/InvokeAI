import { Flex, Text } from '@chakra-ui/react';

import { useState } from 'react';
import {
  MainModelConfigEntity,
  useGetMainModelsQuery,
} from 'services/api/endpoints/models';
import CheckpointModelEdit from './ModelManagerPanel/CheckpointModelEdit';
import DiffusersModelEdit from './ModelManagerPanel/DiffusersModelEdit';
import ModelList from './ModelManagerPanel/ModelList';
import { ALL_BASE_MODELS } from 'services/api/constants';

export default function ModelManagerPanel() {
  const [selectedModelId, setSelectedModelId] = useState<string>();
  const { model } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      model: selectedModelId ? data?.entities[selectedModelId] : undefined,
    }),
  });

  return (
    <Flex sx={{ gap: 8, w: 'full', h: 'full' }}>
      <ModelList
        selectedModelId={selectedModelId}
        setSelectedModelId={setSelectedModelId}
      />
      <ModelEdit model={model} />
    </Flex>
  );
}

type ModelEditProps = {
  model: MainModelConfigEntity | undefined;
};

const ModelEdit = (props: ModelEditProps) => {
  const { model } = props;

  if (model?.model_format === 'checkpoint') {
    return <CheckpointModelEdit key={model.id} model={model} />;
  }

  if (model?.model_format === 'diffusers') {
    return <DiffusersModelEdit key={model.id} model={model} />;
  }

  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        justifyContent: 'center',
        alignItems: 'center',
        maxH: 96,
        userSelect: 'none',
      }}
    >
      <Text variant="subtext">No Model Selected</Text>
    </Flex>
  );
};
