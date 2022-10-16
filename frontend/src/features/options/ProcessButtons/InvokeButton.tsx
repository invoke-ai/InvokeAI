import React from 'react';
import { generateImage, outpaintImage } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton from '../../../common/components/IAIButton';
import { tabMap, tab_dict } from '../../tabs/InvokeTabs';
import useCheckParameters from '../../../common/hooks/useCheckParameters';

export default function InvokeButton() {
  const { activeTab } = useAppSelector(
    (state: RootState) => state.options
  );

  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const handleClickGenerate = () => {
    if (tabMap[activeTab] === 'outpainting') {
      dispatch(outpaintImage());
    }
    else { 
      dispatch(generateImage());
    }
  };

  return (
    <IAIButton
      label="Invoke"
      aria-label="Invoke"
      type="submit"
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn"
    />
  );
}
