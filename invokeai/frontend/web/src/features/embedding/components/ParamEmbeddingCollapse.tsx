import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ParamEmbeddingSelect from './ParamEmbeddingSelect';

export default function ParamEmbeddingCollapse() {
  const shouldShowEmbeddingPicker = useAppSelector(
    (state: RootState) => state.ui.shouldShowEmbeddingPicker
  );

  return (
    shouldShowEmbeddingPicker && (
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ParamEmbeddingSelect />
      </Flex>
    )
  );
}
