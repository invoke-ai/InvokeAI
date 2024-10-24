import { Spacer } from '@invoke-ai/ui-library';
import IAIDroppable from 'common/components/IAIDroppable';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { RegionalGuidanceBadges } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceBadges';
import { RegionalGuidanceSettings } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceSettings';
import { RegionalGuidanceAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { ReplaceLayerImageDropData } from 'features/dnd/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const RegionalGuidance = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'regional_guidance'>>(
    () => ({ id, type: 'regional_guidance' }),
    [id]
  );
  const dropData = useMemo<ReplaceLayerImageDropData>(
    () => ({ id, actionType: 'REPLACE_LAYER_WITH_IMAGE', context: { entityIdentifier } }),
    [id, entityIdentifier]
  );

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <RegionalGuidanceAdapterGate>
        <CanvasEntityContainer>
          <CanvasEntityHeader>
            <CanvasEntityPreviewImage />
            <CanvasEntityEditableTitle />
            <Spacer />
            <RegionalGuidanceBadges />
            <CanvasEntityHeaderCommonActions />
          </CanvasEntityHeader>
          <RegionalGuidanceSettings />
          <IAIDroppable data={dropData} dropLabel={t('controlLayers.replaceLayer')} />
        </CanvasEntityContainer>
      </RegionalGuidanceAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

RegionalGuidance.displayName = 'RegionalGuidance';
