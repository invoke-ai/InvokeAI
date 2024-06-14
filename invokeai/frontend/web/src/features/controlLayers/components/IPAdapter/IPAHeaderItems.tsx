import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { EntityDeleteButton } from 'features/controlLayers/components/LayerCommon/EntityDeleteButton';
import { EntityEnabledToggle } from 'features/controlLayers/components/LayerCommon/EntityEnabledToggle';
import { EntityTitle } from 'features/controlLayers/components/LayerCommon/EntityTitle';
import {
  ipaDeleted,
  ipaIsEnabledToggled,
  selectIPAOrThrow,
} from 'features/controlLayers/store/ipAdaptersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const IPAHeaderItems = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => selectIPAOrThrow(s.ipAdapters, id).isEnabled);
  const onToggle = useCallback(() => {
    dispatch(ipaIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(ipaDeleted({ id }));
  }, [dispatch, id]);

  return (
    <>
      <EntityEnabledToggle isEnabled={isEnabled} onToggle={onToggle} />
      <EntityTitle title={t('controlLayers.ipAdapter')} />
      <Spacer />
      <EntityDeleteButton onDelete={onDelete} />
    </>
  );
});

IPAHeaderItems.displayName = 'IPAHeaderItems';
