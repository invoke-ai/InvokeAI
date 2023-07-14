import { Flex, Text } from '@chakra-ui/react';

import { useState } from 'react';
import {
  MainModelConfigEntity,
  useGetMainModelsQuery,
} from 'services/api/endpoints/models';
import CheckpointModelEdit from './ModelManagerPanel/CheckpointModelEdit';
import DiffusersModelEdit from './ModelManagerPanel/DiffusersModelEdit';
import ModelList from './ModelManagerPanel/ModelList';

export default function ModelManagerPanel() {
  const [selectedModelId, setSelectedModelId] = useState<string>();
  const { model } = useGetMainModelsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      model: selectedModelId ? data?.entities[selectedModelId] : undefined,
    }),
  });

  return (
    <Flex width="100%" columnGap={8}>
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
    return <CheckpointModelEdit model={model} />;
  }

  if (model?.model_format === 'diffusers') {
    return <DiffusersModelEdit model={model} />;
  }

  return (
    <Flex
      sx={{
        width: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 'base',
        bg: 'base.900',
      }}
    >
      <Text fontWeight={500}>Pick A Model To Edit</Text>
    </Flex>
  );
};
