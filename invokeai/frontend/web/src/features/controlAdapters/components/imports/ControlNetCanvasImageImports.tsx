import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { canvasImageToControlAdapter, canvasMaskToControlAdapter } from 'features/canvas/store/actions';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiExcludeBold, PiImageSquareBold } from 'react-icons/pi';

type ControlNetCanvasImageImportsProps = {
  id: string;
};

const ControlNetCanvasImageImports = (props: ControlNetCanvasImageImportsProps) => {
  const { id } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleImportImageFromCanvas = useCallback(() => {
    dispatch(canvasImageToControlAdapter({ id }));
  }, [id, dispatch]);

  const handleImportMaskFromCanvas = useCallback(() => {
    dispatch(canvasMaskToControlAdapter({ id }));
  }, [id, dispatch]);

  return (
    <Flex gap={4}>
      <IconButton
        size="sm"
        icon={<PiImageSquareBold />}
        tooltip={t('controlnet.importImageFromCanvas')}
        aria-label={t('controlnet.importImageFromCanvas')}
        onClick={handleImportImageFromCanvas}
      />
      <IconButton
        size="sm"
        icon={<PiExcludeBold />}
        tooltip={t('controlnet.importMaskFromCanvas')}
        aria-label={t('controlnet.importMaskFromCanvas')}
        onClick={handleImportMaskFromCanvas}
      />
    </Flex>
  );
};

export default memo(ControlNetCanvasImageImports);
