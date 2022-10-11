import React from 'react';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch } from '../../../app/store';
import IAIButton from '../../../common/components/IAIButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';
import { canvasRef } from '../../tabs/Inpainting/InpaintingCanvas';

export default function InvokeButton() {
  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const handleClickGenerate = () => {
    // get dataURL of inpainting canvas
    const maskDataURL = canvasRef?.current?.toDataURL();
    if (maskDataURL) {
      dispatch(
        generateImage({
          inpaintingMask: maskDataURL.split('data:image/png;base64,')[1],
        })
      );
    } else {
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
