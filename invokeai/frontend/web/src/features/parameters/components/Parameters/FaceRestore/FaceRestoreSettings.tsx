import { VStack } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { RootState } from 'app/store/store';
import FaceRestoreType from './FaceRestoreType';
import FaceRestoreStrength from './FaceRestoreStrength';
import CodeformerFidelity from './CodeformerFidelity';

/**
 * Displays face-fixing/GFPGAN options (strength).
 */
const FaceRestoreSettings = () => {
  const facetoolType = useAppSelector(
    (state: RootState) => state.postprocessing.facetoolType
  );

  return (
    <VStack gap={2} alignItems="stretch">
      <FaceRestoreType />
      <FaceRestoreStrength />
      {facetoolType === 'codeformer' && <CodeformerFidelity />}
    </VStack>
  );
};

export default FaceRestoreSettings;
