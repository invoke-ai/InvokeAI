import { Flex, Text, Button } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import SDNumberInput from '../../common/components/SDNumberInput';
import SDSwitch from '../../common/components/SDSwitch';
import randomInt from '../../common/util/randomInt';
import { setSeed, setShouldRandomizeSeed } from './optionsSlice';

/**
 * Seed & variation options. Includes iteration, seed, seed randomization, variation options.
 */
const SeedOptions = () => {
  const { shouldGenerateVariations, shouldRandomizeSeed, seed } =
    useAppSelector((state: RootState) => state.options);

  const dispatch = useAppDispatch();

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  const handleChangeSeed = (v: string | number) => dispatch(setSeed(Number(v)));

  const handleClickRandomizeSeed = () =>
    dispatch(setSeed(randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)));

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
    </Flex>
  );
};

export default SeedOptions;
