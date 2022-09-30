import { FormControl, FormLabel, HStack, Input } from '@chakra-ui/react';
import React, { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import { validateSeedWeights } from '../../../../common/util/seedWeightPairs';
import { setSeedWeights } from '../../optionsSlice';

export default function SeedWeights() {
  const seedWeights = useAppSelector(
    (state: RootState) => state.options.seedWeights
  );

  const shouldGenerateVariations = useAppSelector(
    (state: RootState) => state.options.shouldGenerateVariations
  );

  const dispatch = useAppDispatch();

  const handleChangeSeedWeights = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeedWeights(e.target.value));

  return (
    <FormControl
      isInvalid={
        shouldGenerateVariations &&
        !(validateSeedWeights(seedWeights) || seedWeights === '')
      }
      flexGrow={1}
    >
      <HStack>
        <FormLabel marginInlineEnd={0} marginBottom={1}>
          <p>Seed Weights</p>
        </FormLabel>
        <Input
          size={'sm'}
          value={seedWeights}
          onChange={handleChangeSeedWeights}
          width="12rem"
        />
      </HStack>
    </FormControl>
  );
}
