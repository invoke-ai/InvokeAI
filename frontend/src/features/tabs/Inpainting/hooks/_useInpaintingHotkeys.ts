import { useToast } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

type UseInpaintingHotkeysConfig = {
  activeTab: string;
  brushSize: number;
  handleChangeBrushSize: (newBrushSize: number) => void;
  handleSelectEraserTool: () => void;
  handleSelectBrushTool: () => void;
  canUndo: boolean;
  handleUndo: () => void;
  canRedo: boolean;
  handleRedo: () => void;
  canClearMask: boolean;
  handleClearMask: () => void;
};

const useInpaintingHotkeys = (config: UseInpaintingHotkeysConfig) => {
  const {
    activeTab,
    brushSize,
    handleChangeBrushSize,
    handleSelectEraserTool,
    handleSelectBrushTool,
    canUndo,
    handleUndo,
    canRedo,
    handleRedo,
    canClearMask,
    handleClearMask,
  } = config;

  const toast = useToast();
  // Hotkeys
  useHotkeys(
    '[',
    () => {
      if (activeTab === 'inpainting' && brushSize - 5 > 0) {
        handleChangeBrushSize(brushSize - 5);
      } else {
        handleChangeBrushSize(1);
      }
    },
    [brushSize]
  );

  useHotkeys(
    ']',
    () => {
      if (activeTab === 'inpainting') {
        handleChangeBrushSize(brushSize + 5);
      }
    },
    [brushSize]
  );

  useHotkeys('e', () => {
    if (activeTab === 'inpainting') {
      handleSelectEraserTool();
    }
  });

  useHotkeys('b', () => {
    if (activeTab === 'inpainting') {
      handleSelectBrushTool();
    }
  });

  useHotkeys(
    'cmd+z',
    () => {
      if (activeTab === 'inpainting' && canUndo) {
        handleUndo();
      }
    },
    [canUndo]
  );

  useHotkeys(
    'control+z',
    () => {
      if (activeTab === 'inpainting' && canUndo) {
        handleUndo();
      }
    },
    [canUndo]
  );

  useHotkeys(
    'cmd+shift+z',
    () => {
      if (activeTab === 'inpainting' && canRedo) {
        handleRedo();
      }
    },
    [canRedo]
  );

  useHotkeys(
    'control+shift+z',
    () => {
      if (activeTab === 'inpainting' && canRedo) {
        handleRedo();
      }
    },
    [canRedo]
  );

  useHotkeys(
    'control+y',
    () => {
      if (activeTab === 'inpainting' && canRedo) {
        handleRedo();
      }
    },
    [canRedo]
  );

  useHotkeys(
    'cmd+y',
    () => {
      if (activeTab === 'inpainting' && canRedo) {
        handleRedo();
      }
    },
    [canRedo]
  );

  useHotkeys(
    'c',
    () => {
      if (activeTab === 'inpainting' && canClearMask) {
        handleClearMask();
        toast({
          title: 'Mask Cleared',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [canClearMask]
  );
};

export default useInpaintingHotkeys;
