import { Menu, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { CanvasEntityMenuButton } from 'features/controlLayers/components/common/CanvasEntityMenuButton';
import { imReset } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

export const IMActionsMenu = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onReset = useCallback(() => {
    dispatch(imReset());
  }, [dispatch]);

  return (
    <Menu>
      <CanvasEntityMenuButton />
      <MenuList>
        <MenuItem onClick={onReset} icon={<PiArrowCounterClockwiseBold />}>
          {t('accessibility.reset')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

IMActionsMenu.displayName = 'IMActionsMenu';
