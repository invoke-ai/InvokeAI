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
import { useTranslation } from 'react-i18next';

type ControlNetCanvasImageImportsProps = {
  controlNet: ControlNetConfig;
};

const ControlNetCanvasImageImports = (
  props: ControlNetCanvasImageImportsProps
) => {
  const { controlNet } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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
        tooltip={t('controlnet.importImageFromCanvas')}
        aria-label={t('controlnet.importImageFromCanvas')}
        onClick={handleImportImageFromCanvas}
      />
      <IAIIconButton
        size="sm"
        icon={<FaMask />}
        tooltip={t('controlnet.importMaskFromCanvas')}
        aria-label={t('controlnet.importMaskFromCanvas')}
        onClick={handleImportMaskFromCanvas}
      />
    </Flex>
  );
};

export default memo(ControlNetCanvasImageImports);
