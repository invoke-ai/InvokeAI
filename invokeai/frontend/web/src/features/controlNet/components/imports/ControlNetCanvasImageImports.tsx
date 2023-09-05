import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import {
  canvasImageToControlNet,
  canvasMaskToControlNet,
} from 'features/canvas/store/actions';
import { ControlNetConfig } from 'features/controlNet/store/controlNetSlice';
import { memo, useCallback } from 'react';
import { FaImage, FaMask } from 'react-icons/fa';

type ControlNetCanvasImageImportsProps = {
  controlNet: ControlNetConfig;
};

const ControlNetCanvasImageImports = (
  props: ControlNetCanvasImageImportsProps
) => {
  const { controlNet } = props;
  const dispatch = useAppDispatch();

  const handleImportImageFromCanvas = useCallback(() => {
    dispatch(canvasImageToControlNet({ controlNet }));
  }, [controlNet, dispatch]);

  const handleImportMaskFromCanvas = useCallback(() => {
    dispatch(canvasMaskToControlNet({ controlNet }));
  }, [controlNet, dispatch]);

  return (
    <Flex
      sx={{
        gap: 2,
      }}
    >
      <IAIIconButton
        size="sm"
        icon={<FaImage />}
        tooltip="Import Image From Canvas"
        aria-label="Import Image From Canvas"
        onClick={handleImportImageFromCanvas}
      />
      <IAIIconButton
        size="sm"
        icon={<FaMask />}
        tooltip="Import Mask From Canvas"
        aria-label="Import Mask From Canvas"
        onClick={handleImportMaskFromCanvas}
      />
    </Flex>
  );
};

export default memo(ControlNetCanvasImageImports);
