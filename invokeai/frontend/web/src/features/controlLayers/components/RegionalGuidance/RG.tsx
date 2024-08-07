import { Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/common/CanvasEntityContainer';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { RGActionsMenu } from 'features/controlLayers/components/RegionalGuidance/RGActionsMenu';
import { RGBadges } from 'features/controlLayers/components/RegionalGuidance/RGBadges';
import { RGSettings } from 'features/controlLayers/components/RegionalGuidance/RGSettings';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

import { RGMaskFillColorPicker } from './RGMaskFillColorPicker';
import { RGSettingsPopover } from './RGSettingsPopover';

type Props = {
  id: string;
};

export const RG = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier>(() => ({ id, type: 'regional_guidance' }), [id]);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <CanvasEntityContainer>
        <CanvasEntityHeader onDoubleClick={onToggle}>
          <CanvasEntityEnabledToggle />
          <CanvasEntityTitle />
          <Spacer />
          <RGBadges />
          <RGMaskFillColorPicker />
          <RGSettingsPopover />
          <RGActionsMenu />
          <CanvasEntityDeleteButton />
        </CanvasEntityHeader>
        {isOpen && <RGSettings />}
      </CanvasEntityContainer>
    </EntityIdentifierContext.Provider>
  );
});

RG.displayName = 'RG';
