import ProgressBar from 'features/system/components/ProgressBar';
import SiteHeader from 'features/system/components/SiteHeader';
import Console from 'features/system/components/Console';
import { keepGUIAlive } from './utils';
import InvokeTabs from 'features/tabs/components/InvokeTabs';
import ImageUploader from 'common/components/ImageUploader';

import useToastWatcher from 'features/system/hooks/useToastWatcher';

import FloatingOptionsPanelButtons from 'features/tabs/components/FloatingOptionsPanelButtons';
import FloatingGalleryButton from 'features/tabs/components/FloatingGalleryButton';

keepGUIAlive();

const App = () => {
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
      </ImageUploader>
      <FloatingOptionsPanelButtons />
      <FloatingGalleryButton />
    </div>
  );
};

export default App;
