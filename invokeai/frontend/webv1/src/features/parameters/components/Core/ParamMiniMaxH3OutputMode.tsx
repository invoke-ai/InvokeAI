import { Button, ButtonGroup, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { minimaxH3OutputModeChanged, selectMiniMaxH3OutputMode } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * MiniMax H3 output mode toggle: 'video' (joint audio-video, Generate tab only) or 'image'
 * (a 5-frame minimum clip decoded to one gallery image - the txt2img mode, canvas-capable).
 */
const ParamMiniMaxH3OutputMode = () => {
  const { t } = useTranslation();
  const outputMode = useAppSelector(selectMiniMaxH3OutputMode);
  const dispatch = useAppDispatch();

  const onClickVideo = useCallback(() => dispatch(minimaxH3OutputModeChanged('video')), [dispatch]);
  const onClickImage = useCallback(() => dispatch(minimaxH3OutputModeChanged('image')), [dispatch]);

  return (
    <FormControl>
      <FormLabel>{t('parameters.minimaxH3OutputMode')}</FormLabel>
      <ButtonGroup size="sm" isAttached variant="outline">
        <Button onClick={onClickVideo} colorScheme={outputMode === 'video' ? 'invokeBlue' : 'base'}>
          {t('parameters.minimaxH3OutputModeVideo')}
        </Button>
        <Button onClick={onClickImage} colorScheme={outputMode === 'image' ? 'invokeBlue' : 'base'}>
          {t('parameters.minimaxH3OutputModeImage')}
        </Button>
      </ButtonGroup>
    </FormControl>
  );
};

export default memo(ParamMiniMaxH3OutputMode);
