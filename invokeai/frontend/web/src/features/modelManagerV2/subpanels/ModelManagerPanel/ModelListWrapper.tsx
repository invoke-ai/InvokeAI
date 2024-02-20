import { Flex } from '@invoke-ai/ui-library';
import type { AnyModelConfig } from 'services/api/types';

import { ModelListHeader } from './ModelListHeader';
import ModelListItem from './ModelListItem';

type ModelListWrapperProps = {
  title: string;
  modelList: AnyModelConfig[];
};

export const ModelListWrapper = (props: ModelListWrapperProps) => {
  const { title, modelList } = props;
  return (
    <Flex flexDirection="column" p="10px 0">
      <Flex gap={2} flexDir="column">
        <ModelListHeader title={title} />

        {modelList.map((model) => (
          <ModelListItem key={model.key} model={model} />
        ))}
      </Flex>
    </Flex>
  );
};
