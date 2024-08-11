import { Box, Flex, Heading } from '@invoke-ai/ui-library';
import { memo } from 'react';
import type { AnyModelConfig } from 'services/api/types';

import ModelListItem from './ModelListItem';

type ModelListWrapperProps = {
  title: string;
  modelList: AnyModelConfig[];
};

export const ModelListWrapper = memo((props: ModelListWrapperProps) => {
  const { title, modelList } = props;
  return (
    <Box>
      <Box pb={2} position="sticky" zIndex={1} top={0} bg="base.900">
        <Heading size="sm">{title}</Heading>
      </Box>
      <Flex flexDir="column" gap={1}>
        {modelList.map((model) => (
          <ModelListItem key={model.key} model={model} />
        ))}
      </Flex>
    </Box>
  );
});

ModelListWrapper.displayName = 'ModelListWrapper';
