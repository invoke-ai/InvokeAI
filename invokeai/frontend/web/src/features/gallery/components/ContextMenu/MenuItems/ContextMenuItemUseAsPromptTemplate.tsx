import { MenuItem } from '@invoke-ai/ui-library';
import {  useItemDTOContextImageOnly } from 'features/gallery/contexts/ItemDTOContext';
import { useCreateStylePresetFromMetadata } from 'features/gallery/hooks/useCreateStylePresetFromMetadata';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPaintBrushBold } from 'react-icons/pi';
import type { ImageDTO } from 'services/api/types';

export const ContextMenuItemUseAsPromptTemplate = memo(() => {
  const { t } = useTranslation();
  const imageDTO = useItemDTOContextImageOnly();

  const stylePreset = useCreateStylePresetFromMetadata(imageDTO);

  return (
    <MenuItem icon={<PiPaintBrushBold />} onClickCapture={stylePreset.create} isDisabled={!stylePreset.isEnabled}>
      {t('stylePresets.useForTemplate')}
    </MenuItem>
  );
});

ContextMenuItemUseAsPromptTemplate.displayName = 'ContextMenuItemUseAsPromptTemplate';
