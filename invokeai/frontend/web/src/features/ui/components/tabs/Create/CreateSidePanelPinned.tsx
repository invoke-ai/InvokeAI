import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import { Panel } from 'react-resizable-panels';
import CreateTextParameters from './CreateBaseSettings';
import PinParametersPanelButton from '../../PinParametersPanelButton';
import ResizeHandle from '../ResizeHandle';
import ImageToImageSettings from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageSettings';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import CreateImageSettings from './CreateImageSettings';

const selector = createSelector(
  uiSelector,
  (ui) => {
    const {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      shouldShowImageParameters,
    } = ui;

    return {
      shouldPinParametersPanel,
      shouldShowParametersPanel,
      shouldShowImageParameters,
    };
  },
  defaultSelectorOptions
);

const CreateSidePanelPinned = () => {
  const dispatch = useAppDispatch();
  const {
    shouldPinParametersPanel,
    shouldShowParametersPanel,
    shouldShowImageParameters,
  } = useAppSelector(selector);
  return (
    <>
      <Panel
        order={0}
        defaultSize={25}
        minSize={25}
        style={{ position: 'relative' }}
      >
        <CreateTextParameters />
        <PinParametersPanelButton
          sx={{ position: 'absolute', top: 0, insetInlineEnd: 0 }}
        />
      </Panel>
      {shouldShowImageParameters && (
        <>
          <ResizeHandle />
          <Panel
            order={1}
            defaultSize={25}
            minSize={25}
            style={{ position: 'relative' }}
          >
            <CreateImageSettings />
          </Panel>
        </>
      )}
      <ResizeHandle />
    </>
  );
};

export default memo(CreateSidePanelPinned);
