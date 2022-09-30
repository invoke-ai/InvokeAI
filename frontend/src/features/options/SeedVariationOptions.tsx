import {
  Flex,
  Input,
  HStack,
  FormControl,
  FormLabel,
  Text,
  Button,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import { ChangeEvent } from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSwitch from '../../common/components/SDSwitch';
import randomInt from '../../common/util/randomInt';
import { validateSeedWeights } from '../../common/util/seedWeightPairs';
import {
  OptionsState,
  setSeed,
  setSeedWeights,
  setShouldGenerateVariations,
  setShouldRandomizeSeed,
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
const SeedVariationOptions = () => {
  const {
    shouldGenerateVariations,
    variationAmount,
    seedWeights,
    shouldRandomizeSeed,
    seed,
  } = useAppSelector(optionsSelector);

  const dispatch = useAppDispatch();

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  const handleChangeSeed = (v: number) => dispatch(setSeed(v));

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)));

  const handleChangeShouldGenerateVariations = (
    e: ChangeEvent<HTMLInputElement>
  ) => dispatch(setShouldGenerateVariations(e.target.checked));

  const handleChangevariationAmount = (v: number) =>
    dispatch(setVariationAmount(v));

  const handleChangeSeedWeights = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeedWeights(e.target.value));

  return (
    <Flex gap={2} direction={'column'}>
      <SDSwitch
        label="Randomize Seed"
        isChecked={shouldRandomizeSeed}
        onChange={handleChangeShouldRandomizeSeed}
      />
      <Flex gap={2}>
        <SDNumberInput
          label="Seed"
          step={1}
          precision={0}
          flexGrow={1}
          min={NUMPY_RAND_MIN}
          max={NUMPY_RAND_MAX}
          isDisabled={shouldRandomizeSeed}
          isInvalid={seed < 0 && shouldGenerateVariations}
          onChange={handleChangeSeed}
          value={seed}
        />
        <Button
          size={'sm'}
          isDisabled={shouldRandomizeSeed}
          onClick={handleClickRandomizeSeed}
        >
          <Text pl={2} pr={2}>
            Shuffle
          </Text>
        </Button>
      </Flex>
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

export default SeedVariationOptions;
