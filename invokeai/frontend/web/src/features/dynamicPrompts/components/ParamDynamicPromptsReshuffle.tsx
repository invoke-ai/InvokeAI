import { Button, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { randomSeedChanged, selectDynamicPromptsMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';

const ParamDynamicPromptsReshuffle = () => {
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);

  const reshuffle = useCallback(() => {
    dispatch(randomSeedChanged(Date.now()));
  }, [dispatch]);

  if (mode !== 'random') {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>Preview</FormLabel>
      <Button onClick={reshuffle} variant="outline">
        Reshuffle Now
      </Button>
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsReshuffle);
