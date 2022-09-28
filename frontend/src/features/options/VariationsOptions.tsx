import {
  Flex,
  Input,
  HStack,
  FormControl,
  FormLabel,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';

import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSwitch from '../../common/components/SDSwitch';

import { validateSeedWeights } from '../../common/util/seedWeightPairs';
import {
  OptionsState,
  setSeedWeights,
  setShouldGenerateVariations,
  setVariationAmount,
} from './optionsSlice';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      variationAmount: options.variationAmount,
      seedWeights: options.seedWeights,
      shouldGenerateVariations: options.shouldGenerateVariations,
      shouldRandomizeSeed: options.shouldRandomizeSeed,
      seed: options.seed,
      iterations: options.iterations,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const VariationsOptions = () => {
  const { shouldGenerateVariations, variationAmount, seedWeights } =
    useAppSelector(optionsSelector);

  const dispatch = useAppDispatch();

  const handleChangeShouldGenerateVariations = (
    e: ChangeEvent<HTMLInputElement>
  ) => dispatch(setShouldGenerateVariations(e.target.checked));

  const handleChangevariationAmount = (v: string | number) =>
    dispatch(setVariationAmount(Number(v)));

  const handleChangeSeedWeights = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeedWeights(e.target.value));

  return (
    <Flex gap={2} direction={'column'}>
      <SDSwitch
        label="Generate Variations"
        isChecked={shouldGenerateVariations}
        width={'auto'}
        onChange={handleChangeShouldGenerateVariations}
      />
      <SDNumberInput
        label="Variation Amount"
        value={variationAmount}
        step={0.01}
        min={0}
        max={1}
        isDisabled={!shouldGenerateVariations}
        onChange={handleChangevariationAmount}
        width="90px"
      />
      <FormControl
        isInvalid={
          shouldGenerateVariations &&
          !(validateSeedWeights(seedWeights) || seedWeights === '')
        }
        flexGrow={1}
      >
        <HStack>
          <FormLabel marginInlineEnd={0} marginBottom={1}>
            <Text whiteSpace="nowrap">Seed Weights</Text>
          </FormLabel>
          <Input
            size={'sm'}
            value={seedWeights}
            onChange={handleChangeSeedWeights}
          />
        </HStack>
      </FormControl>
    </Flex>
  );
};

export default VariationsOptions;
