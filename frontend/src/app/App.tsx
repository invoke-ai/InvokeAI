import { useEffect } from 'react';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import Console from '../features/system/Console';
import { useAppDispatch } from './store';
import { requestSystemConfig } from './socketio/actions';
import { keepGUIAlive } from './utils';
import InvokeTabs from '../features/tabs/InvokeTabs';
import ImageUploader from '../common/components/ImageUploader';
import { RootState, useAppSelector } from '../app/store';

import FloatingGalleryButton from '../features/tabs/FloatingGalleryButton';
import FloatingOptionsPanelButtons from '../features/tabs/FloatingOptionsPanelButtons';
import { createSelector } from '@reduxjs/toolkit';
import { GalleryState } from '../features/gallery/gallerySlice';
import { OptionsState } from '../features/options/optionsSlice';
import { activeTabNameSelector } from '../features/options/optionsSelectors';
import { SystemState } from '../features/system/systemSlice';
import _ from 'lodash';
import { Model } from './invokeai';

keepGUIAlive();

const appSelector = createSelector(
  [
    (state: RootState) => state.gallery,
    (state: RootState) => state.options,
    (state: RootState) => state.system,
    activeTabNameSelector,
  ],
  (
    gallery: GalleryState,
    options: OptionsState,
    system: SystemState,
    activeTabName
  ) => {
    const { shouldShowGallery, shouldHoldGalleryOpen, shouldPinGallery } =
      gallery;
    const {
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
      shouldPinOptionsPanel,
    } = options;

    const modelStatusText = _.reduce(
      system.model_list,
      (acc: string, cur: Model, key: string) => {
        if (cur.status === 'active') acc = key;
        return acc;
      },
      ''
    );

    const shouldShowGalleryButton = !(
      shouldShowGallery ||
      (shouldHoldGalleryOpen && !shouldPinGallery)
    );

    const shouldShowOptionsPanelButton =
      !(
        shouldShowOptionsPanel ||
        (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
      ) && ['txt2img', 'img2img', 'inpainting'].includes(activeTabName);

    return {
      modelStatusText,
      shouldShowGalleryButton,
      shouldShowOptionsPanelButton,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const App = () => {
  const dispatch = useAppDispatch();

  const { shouldShowGalleryButton, shouldShowOptionsPanelButton } =
    useAppSelector(appSelector);

  useEffect(() => {
    dispatch(requestSystemConfig());
  }, [dispatch]);

  return (
    <div className="App">
      <ImageUploader>
        <ProgressBar />
        <div className="app-content">
          <SiteHeader />
          <InvokeTabs />
        </div>
        <div className="app-console">
          <Console />
        </div>
        {shouldShowGalleryButton && <FloatingGalleryButton />}
        {shouldShowOptionsPanelButton && <FloatingOptionsPanelButtons />}
      </ImageUploader>
    </div>
  );
};

export default App;
