import { useState } from 'react';
import { FaMask } from 'react-icons/fa';

import IAIPopover from 'common/components/IAIPopover';
import IAIIconButton from 'common/components/IAIIconButton';

import IAICanvasMaskInvertControl from './IAICanvasMaskControls/IAICanvasMaskInvertControl';
import IAICanvasMaskVisibilityControl from './IAICanvasMaskControls/IAICanvasMaskVisibilityControl';
import IAICanvasMaskColorPicker from './IAICanvasMaskControls/IAICanvasMaskColorPicker';

export default function IAICanvasMaskControl() {
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
          <IAICanvasMaskVisibilityControl />
          <IAICanvasMaskInvertControl />
          <IAICanvasMaskColorPicker />
        </div>
      </IAIPopover>
    </>
  );
}
