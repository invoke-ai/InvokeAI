import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { ipaDeleted, ipaIsEnabledToggled, selectIPAOrThrow } from 'features/controlLayers/store/ipAdaptersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  onToggleVisibility: () => void;
};

export const IPAHeader = memo(({ id, onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => selectIPAOrThrow(s.ipAdapters, id).isEnabled);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(ipaIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(ipaDeleted({ id }));
  }, [dispatch, id]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={t('controlLayers.ipAdapter')} />
      <Spacer />
      <CanvasEntityDeleteButton onDelete={onDelete} />
    </CanvasEntityHeader>
  );
});

IPAHeader.displayName = 'IPAHeader';
