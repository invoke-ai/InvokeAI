import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import Console from 'features/system/components/Console';
import { keepGUIAlive } from './utils';
import InvokeTabs from 'features/tabs/components/InvokeTabs';
import ImageUploader from 'common/components/ImageUploader';
import { RootState, useAppSelector } from 'app/store';

import FloatingGalleryButton from 'features/tabs/components/FloatingGalleryButton';
import FloatingOptionsPanelButtons from 'features/tabs/components/FloatingOptionsPanelButtons';
import { createSelector } from '@reduxjs/toolkit';
import { GalleryState } from 'features/gallery/store/gallerySlice';
import { OptionsState } from 'features/options/store/optionsSlice';
import { activeTabNameSelector } from 'features/options/store/optionsSelectors';
import { SystemState } from 'features/system/store/systemSlice';
import _ from 'lodash';
import { Model } from './invokeai';
import useToastWatcher from 'features/system/hooks/useToastWatcher';

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

    const shouldShowGalleryButton =
      !(shouldShowGallery || (shouldHoldGalleryOpen && !shouldPinGallery)) &&
      ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

    const shouldShowOptionsPanelButton =
      !(
        shouldShowOptionsPanel ||
        (shouldHoldOptionsPanelOpen && !shouldPinOptionsPanel)
      ) && ['txt2img', 'img2img', 'unifiedCanvas'].includes(activeTabName);

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
  const { shouldShowGalleryButton, shouldShowOptionsPanelButton } =
    useAppSelector(appSelector);

  useToastWatcher();

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
