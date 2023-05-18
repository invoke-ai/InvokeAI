import { useCallback } from 'react';

let openUploader = () => {
  return;
};

const useImageUploader = () => {
  const setOpenUploaderFunction = useCallback(
    (openUploaderFunction?: () => void) => {
      if (openUploaderFunction) {
        openUploader = openUploaderFunction;
      }
    },
    []
  );

  return {
    setOpenUploaderFunction,
    openUploader,
  };
};

export default useImageUploader;
