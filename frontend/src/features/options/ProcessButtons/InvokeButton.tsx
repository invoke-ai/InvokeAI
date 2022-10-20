import React from 'react';
import { generateImage, outpaintImage } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton from '../../../common/components/IAIButton';
import { tabMap, tab_dict } from '../../tabs/InvokeTabs';
import useCheckParameters from '../../../common/hooks/useCheckParameters';
import { canvasRef } from '../../gallery/Canvas/PaintingCanvas';

export default function InvokeButton() {
  const { activeTab, width, height } = useAppSelector(
    (state: RootState) => state.options
  );

  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const handleClickGenerate = () => {
    if (!canvasRef.current) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    if (tempCtx) {
      tempCtx.drawImage(canvasRef.current, 0, 0);
    }
    const maskDataURLString = tempCanvas.toDataURL();

    if (maskDataURLString) {
      console.log('maskDataURLString', maskDataURLString);
    
      dispatch(
        generateImage({
          inpaintingMask: maskDataURLString.split('data:image/png;base64,')[1],
        })
      );
    } else {
      if (tabMap[activeTab] === 'outpainting') {
        dispatch(outpaintImage());
      }
      else {
        dispatch(generateImage());
      }
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
