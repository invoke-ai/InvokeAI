import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { activeTabNameSelector } from '../store/uiSelectors';
import TextTabParametersDrawer from './tabs/text/TextTabParametersDrawer';
import { RootState } from 'app/store/store';

const ParametersDrawer = () => {
  const activeTabName = useAppSelector(activeTabNameSelector);
  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  if (shouldPinParametersPanel) {
    return null;
  }

  if (activeTabName === 'text') {
    return <TextTabParametersDrawer />;
  }

  if (activeTabName === 'image') {
    return null;
  }

  if (activeTabName === 'unifiedCanvas') {
    return null;
  }

  return null;
};

export default memo(ParametersDrawer);
