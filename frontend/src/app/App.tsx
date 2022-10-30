import { useEffect, useState } from 'react';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import Console from '../features/system/Console';
import Loading from '../Loading';
import { useAppDispatch } from './store';
import { requestSystemConfig } from './socketio/actions';
import { keepGUIAlive } from './utils';
import InvokeTabs, { tabMap } from '../features/tabs/InvokeTabs';
import ImageUploader from '../common/components/ImageUploader';
import { RootState, useAppSelector } from '../app/store';

import ShowHideGalleryButton from '../features/tabs/ShowHideGalleryButton';
import ShowHideOptionsPanelButton from '../features/tabs/ShowHideOptionsPanelButton';
import { createSelector } from '@reduxjs/toolkit';
import { GalleryState } from '../features/gallery/gallerySlice';
import { OptionsState } from '../features/options/optionsSlice';

keepGUIAlive();

const appSelector = createSelector(
  [(state: RootState) => state.gallery, (state: RootState) => state.options],
  (gallery: GalleryState, options: OptionsState) => {
    const { shouldShowGallery, shouldHoldGalleryOpen, shouldPinGallery } =
      gallery;
    const {
      shouldShowOptionsPanel,
      shouldHoldOptionsPanelOpen,
      shouldPinOptionsPanel,
      activeTab,
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
        ) && ['txt2img', 'img2img', 'inpainting'].includes(tabMap[activeTab]),
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
        {shouldShowGalleryButton && <ShowHideGalleryButton />}
        {shouldShowOptionsPanelButton && <ShowHideOptionsPanelButton />}
      </ImageUploader>
    </div>
  ) : (
    <Loading />
  );
};

export default App;
