import React, { useState } from 'react';
import { FaMask } from 'react-icons/fa';

import IAIIconButton from '../../../../common/components/IAIIconButton';
import IAIPopover from '../../../../common/components/IAIPopover';

import InpaintingMaskVisibilityControl from './InpaintingMaskControls/InpaintingMaskVisibilityControl';
import InpaintingMaskInvertControl from './InpaintingMaskControls/InpaintingMaskInvertControl';
import InpaintingMaskColorPicker from './InpaintingMaskControls/InpaintingMaskColorPicker';

export default function InpaintingMaskControl() {
  const [maskOptionsOpen, setMaskOptionsOpen] = useState<boolean>(false);

  return (
    <>
      <IAIPopover
        trigger="hover"
        onOpen={() => setMaskOptionsOpen(true)}
        onClose={() => setMaskOptionsOpen(false)}
        triggerComponent={
          <IAIIconButton
            aria-label="Mask Options"
            tooltip="Mask Options"
            icon={<FaMask />}
            cursor={'pointer'}
            data-selected={maskOptionsOpen}
          />
        }
      >
        <div className="inpainting-button-dropdown">
          <InpaintingMaskVisibilityControl />
          <InpaintingMaskInvertControl />
          <InpaintingMaskColorPicker />
        </div>
      </IAIPopover>
    </>
  );
}
