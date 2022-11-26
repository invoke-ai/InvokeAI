let openFunction: () => void;

const useImageUploader = () => {
  return {
    setOpenUploader: (open?: () => void) => {
      if (open) {
        openFunction = open;
      }
    },
    openUploader: openFunction,
  };
};

export default useImageUploader;
