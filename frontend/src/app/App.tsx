import { useEffect, useState } from 'react';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import Console from '../features/system/Console';
import Loading from '../Loading';
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

keepGUIAlive();

const appSelector = createSelector(
  [
    (state: RootState) => state.gallery,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (gallery: GalleryState, options: OptionsState, activeTabName) => {
    const { shouldShowGallery, shouldHoldGalleryOpen, shouldPinGallery } =
      gallery;
    const {
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
      shouldPinOptionsPanel,
    } = options;

    return {
      shouldShowGalleryButton: !(
        shouldShowGallery ||
        (shouldHoldGalleryOpen && !shouldPinGallery)
      ),
      shouldShowOptionsPanelButton:
        !(
          shouldShowOptionsPanel ||
          (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
        ) && ['txt2img', 'img2img', 'inpainting'].includes(activeTabName),
    };
  }
);

const App = () => {
  const dispatch = useAppDispatch();

  const [isReady, setIsReady] = useState<boolean>(false);

  const { shouldShowGalleryButton, shouldShowOptionsPanelButton } =
    useAppSelector(appSelector);

  useEffect(() => {
    dispatch(requestSystemConfig());
    setIsReady(true);
  }, [dispatch]);

  return isReady ? (
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
  ) : (
    <Loading />
  );
};

export default App;
