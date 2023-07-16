import { useState } from 'react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { Panel } from 'reactflow';
import FieldTypeLegend from '../FieldTypeLegend';
import NodeGraphOverlay from '../NodeGraphOverlay';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaArrowUp, FaArrowDown } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';

const TopRightPanel = () => {
  const { t } = useTranslation();
  const [isLegendVisible, setIsLegendVisible] = useState(true);
  const shouldShowGraphOverlay = useAppSelector(
    (state: RootState) => state.nodes.shouldShowGraphOverlay
  );

  const toggleLegendVisibility = () => {
    setIsLegendVisible(!isLegendVisible);
  };

  return (
    <Panel position="top-right">
      <div style={{ position: 'relative' }}>
        <IAIIconButton
          onClick={toggleLegendVisibility}
          isChecked={isLegendVisible}
          tooltip={
            isLegendVisible ? t('nodes.closeLegend') : t('nodes.openLegend')
          }
        >
          {isLegendVisible ? <FaArrowUp /> : <FaArrowDown />}
        </IAIIconButton>
        {isLegendVisible && (
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 5px)',
              right: 0,
              zIndex: 1,
            }}
          >
            <FieldTypeLegend />
          </div>
        )}
      </div>
      {shouldShowGraphOverlay && <NodeGraphOverlay />}
    </Panel>
  );
};

export default memo(TopRightPanel);
