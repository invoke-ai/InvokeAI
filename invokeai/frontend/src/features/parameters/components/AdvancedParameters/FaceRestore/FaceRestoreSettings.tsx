import { Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/storeHooks';
import type { RootState } from 'app/store';
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
    <Flex direction="column" gap={2} minWidth="20rem">
      <FaceRestoreType />
      <FaceRestoreStrength />
      {facetoolType === 'codeformer' && <CodeformerFidelity />}
    </Flex>
  );
};

export default FaceRestoreSettings;
