import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { sessionModeChanged } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const CanvasModeSwitcher = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const mode = useAppSelector((s) => s.canvasV2.session.mode);
  const onClickGenerate = useCallback(() => dispatch(sessionModeChanged({ mode: 'generate' })), [dispatch]);
  const onClickCompose = useCallback(() => dispatch(sessionModeChanged({ mode: 'compose' })), [dispatch]);

  return (
    <ButtonGroup variant="outline">
      <Button onClick={onClickGenerate} colorScheme={mode === 'generate' ? 'invokeBlue' : 'base'}>
        {t('controlLayers.generateMode')}
      </Button>
      <Button onClick={onClickCompose} colorScheme={mode === 'compose' ? 'invokeBlue' : 'base'}>
        {t('controlLayers.composeMode')}
      </Button>
    </ButtonGroup>
  );
});

CanvasModeSwitcher.displayName = 'CanvasModeSwitcher';
