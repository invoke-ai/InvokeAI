import { useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { IPAHeader } from 'features/controlLayers/components/IPAdapter/IPAHeader';
import { IPASettings } from 'features/controlLayers/components/IPAdapter/IPASettings';
import { entitySelected } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const IPA = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onSelect = useCallback(() => {
    dispatch(entitySelected({ id, type: 'ip_adapter' }));
  }, [dispatch, id]);

  return (
    <CanvasEntityContainer isSelected={isSelected} onSelect={onSelect}>
      <IPAHeader id={id} onToggleVisibility={onToggle} />
      {isOpen && <IPASettings id={id} />}
    </CanvasEntityContainer>
  );
});

IPA.displayName = 'IPA';
