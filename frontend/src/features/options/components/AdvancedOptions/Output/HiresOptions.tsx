import { Flex } from '@chakra-ui/react';
import { ChangeEvent } from 'react';
import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from 'app/store';
import IAISwitch from 'common/components/IAISwitch';
import { setHiresFix } from 'features/options/store/optionsSlice';

/**
 * Hires Fix Toggle
 */
const HiresOptions = () => {
  const dispatch = useAppDispatch();

  const hiresFix = useAppSelector((state: RootState) => state.options.hiresFix);

  const handleChangeHiresFix = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setHiresFix(e.target.checked));

  return (
    <Flex gap={2} direction={'column'}>
      <IAISwitch
        label="High Res Optimization"
        fontSize={'md'}
        isChecked={hiresFix}
        onChange={handleChangeHiresFix}
      />
    </Flex>
  );
};

export default HiresOptions;
