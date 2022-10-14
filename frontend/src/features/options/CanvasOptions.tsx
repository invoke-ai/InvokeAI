import { Flex } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { setCanvasHeight, setCanvasWidth } from './optionsSlice';
import { ChangeEvent } from 'react';
import IAINumberInput from '../../common/components/IAINumberInput';

/**
 * Canvas size options. Includes width, height.
 */
const CanvasOptions = () => {
  const dispatch = useAppDispatch();

  const { canvasWidth, canvasHeight } = useAppSelector((state: RootState) => state.options);

  const handleChangeCanvasWidth = (t: number) => 
    dispatch(setCanvasWidth(t));
  const handleChangeCanvasHeight = (t: number) =>
    dispatch(setCanvasHeight(t));
  

  return (
    <Flex gap={2} direction={'column'}>
      <IAINumberInput
        value={canvasWidth ?? 1024}
        label='Width'
        min={1}
        max={16384}
        step={1}
        onChange={handleChangeCanvasWidth}
      />
      <IAINumberInput
        value={canvasHeight ?? 1024}
        label='Height'
        min={1}
        max={16384}
        step={1}
        onChange={handleChangeCanvasHeight}
      />
    </Flex>
  );
};

export default CanvasOptions;
