import { Flex } from '@chakra-ui/layout';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setFacetool, setShouldRunGFPGAN } from '../../optionsSlice';

export default function FaceRestore() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const isCodeformerAvailable = useAppSelector(
    (state: RootState) => state.system.isCodeformerAvailable
  );

  const shouldRunGFPGAN = useAppSelector(
    (state: RootState) => state.options.shouldRunGFPGAN
  );

  const shouldRunCodeformer = useAppSelector(
    (state: RootState) => state.options.shouldRunCodeformer
  );

  const facetool = useAppSelector(
    (state: RootState) => state.options.facetool
  );


  const dispatch = useAppDispatch();

  const handleChangeShouldRunGFPGAN = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunGFPGAN(e.target.checked));

  const handleChangeShouldRunCodeformer = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunCodeformer(e.target.checked));

  const handleChangeFacetool = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setFacetool(e.target.value));

  return (
    <Flex
      justifyContent={'space-between'}
      alignItems={'center'}
      width={'100%'}
      mr={2}
    > 
      <p>Restore Face</p>
      <IAISwitch
        isDisabled={!isGFPGANAvailable}
        isChecked={shouldRunGFPGAN}
        onChange={handleChangeShouldRunGFPGAN}
      />
    </Flex>
  );
}
/** TO DO -- 
 * Update switch logic to check for both GFPGAN and codeformer
 * Remove "shouldrun" and push lower into the facetool selector (if facetool = gfpgan, shouldRunGFPGAN)
 * onChange should also be pushed down.
 */