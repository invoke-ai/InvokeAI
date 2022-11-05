import { Flex } from '@chakra-ui/layout';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import IAISwitch from '../../../../common/components/IAISwitch';
import { setShouldRunFacetool } from '../../optionsSlice';

export default function FaceRestoreHeader() {
  const isGFPGANAvailable = useAppSelector(
    (state: RootState) => state.system.isGFPGANAvailable
  );

  const shouldRunFacetool = useAppSelector(
    (state: RootState) => state.options.shouldRunFacetool
  );

  const dispatch = useAppDispatch();

  const handleChangeShouldRunFacetool = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRunFacetool(e.target.checked));

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
        isChecked={shouldRunFacetool}
        onChange={handleChangeShouldRunFacetool}
      />
    </Flex>
  );
}
