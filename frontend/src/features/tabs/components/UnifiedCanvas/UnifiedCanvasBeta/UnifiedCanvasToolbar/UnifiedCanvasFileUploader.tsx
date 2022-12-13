import { useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import useImageUploader from 'common/hooks/useImageUploader';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import React from 'react';
import { FaUpload } from 'react-icons/fa';

export default function UnifiedCanvasFileUploader() {
  const isStaging = useAppSelector(isStagingSelector);
  const { openUploader } = useImageUploader();

  return (
    <IAIIconButton
      aria-label="Upload"
      tooltip="Upload"
      icon={<FaUpload />}
      onClick={openUploader}
      isDisabled={isStaging}
    />
  );
}
