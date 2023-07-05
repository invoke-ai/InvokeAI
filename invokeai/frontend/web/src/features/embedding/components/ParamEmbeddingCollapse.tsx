import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import ParamEmbeddingSelect from './ParamEmbeddingSelect';

export default function ParamEmbeddingCollapse() {
  return (
    <IAICollapse label="Embeddings">
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamEmbeddingSelect />
      </Flex>
    </IAICollapse>
  );
}
