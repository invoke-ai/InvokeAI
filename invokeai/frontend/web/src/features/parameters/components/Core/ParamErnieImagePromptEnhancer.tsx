import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectErnieImageUsePromptEnhancer,
  setErnieImageUsePromptEnhancer,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

const ParamErnieImagePromptEnhancer = () => {
  const dispatch = useAppDispatch();
  const useEnhancer = useAppSelector(selectErnieImageUsePromptEnhancer);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setErnieImageUsePromptEnhancer(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>Prompt Enhancer</FormLabel>
      <Switch isChecked={useEnhancer} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamErnieImagePromptEnhancer);
