import { Menu, MenuItem, MenuList, Spacer } from '@invoke-ai/ui-library';
import { createAppSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CAOpacityAndFilter } from 'features/controlLayers/components/ControlAdapter/CAOpacityAndFilter';
import { EntityDeleteButton } from 'features/controlLayers/components/LayerCommon/EntityDeleteButton';
import { EntityEnabledToggle } from 'features/controlLayers/components/LayerCommon/EntityEnabledToggle';
import { EntityMenuButton } from 'features/controlLayers/components/LayerCommon/EntityMenuButton';
import { EntityTitle } from 'features/controlLayers/components/LayerCommon/EntityTitle';
import {
  caDeleted,
  caIsEnabledToggled,
  caMovedBackwardOne,
  caMovedForwardOne,
  caMovedToBack,
  caMovedToFront,
  selectCA,
  selectControlAdaptersV2Slice,
} from 'features/controlLayers/store/controlAdaptersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowDownBold,
  PiArrowLineDownBold,
  PiArrowLineUpBold,
  PiArrowUpBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  id: string;
};

const selectValidActions = createAppSelector(
  [selectControlAdaptersV2Slice, (caState, id: string) => id],
  (caState, id) => {
    const ca = selectCA(caState, id);
    assert(ca, `CA with id ${id} not found`);
    const caIndex = caState.controlAdapters.indexOf(ca);
    const caCount = caState.controlAdapters.length;
    return {
      canMoveForward: caIndex < caCount - 1,
      canMoveBackward: caIndex > 0,
      canMoveToFront: caIndex < caCount - 1,
      canMoveToBack: caIndex > 0,
    };
  }
);

export const CAHeaderItems = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const validActions = useAppSelector((s) => selectValidActions(s, id));
  const isEnabled = useAppSelector((s) => {
    const ca = selectCA(s.controlAdaptersV2, id);
    assert(ca, `CA with id ${id} not found`);
    return ca.isEnabled;
  });
  const onToggle = useCallback(() => {
    dispatch(caIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(caDeleted({ id }));
  }, [dispatch, id]);
  const moveForwardOne = useCallback(() => {
    dispatch(caMovedForwardOne({ id }));
  }, [dispatch, id]);
  const moveToFront = useCallback(() => {
    dispatch(caMovedToFront({ id }));
  }, [dispatch, id]);
  const moveBackwardOne = useCallback(() => {
    dispatch(caMovedBackwardOne({ id }));
  }, [dispatch, id]);
  const moveToBack = useCallback(() => {
    dispatch(caMovedToBack({ id }));
  }, [dispatch, id]);

  return (
    <>
      <EntityEnabledToggle isEnabled={isEnabled} onToggle={onToggle} />
      <EntityTitle title={t('controlLayers.globalControlAdapter')} />
      <Spacer />
      <CAOpacityAndFilter id={id} />
      <Menu>
        <EntityMenuButton />
        <MenuList>
          <MenuItem onClick={moveToFront} isDisabled={!validActions.canMoveToFront} icon={<PiArrowLineUpBold />}>
            {t('controlLayers.moveToFront')}
          </MenuItem>
          <MenuItem onClick={moveForwardOne} isDisabled={!validActions.canMoveForward} icon={<PiArrowUpBold />}>
            {t('controlLayers.moveForward')}
          </MenuItem>
          <MenuItem onClick={moveBackwardOne} isDisabled={!validActions.canMoveBackward} icon={<PiArrowDownBold />}>
            {t('controlLayers.moveBackward')}
          </MenuItem>
          <MenuItem onClick={moveToBack} isDisabled={!validActions.canMoveToBack} icon={<PiArrowLineDownBold />}>
            {t('controlLayers.moveToBack')}
          </MenuItem>
          <MenuItem onClick={onDelete} icon={<PiTrashSimpleBold />} color="error.300">
            {t('common.delete')}
          </MenuItem>
        </MenuList>
      </Menu>
      <EntityDeleteButton onDelete={onDelete} />
    </>
  );
});

CAHeaderItems.displayName = 'CAHeaderItems';
