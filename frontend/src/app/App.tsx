import { useEffect, useState } from 'react';
import CurrentImageDisplay from '../features/gallery/CurrentImageDisplay';
import ImageGallery from '../features/gallery/ImageGallery';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import LogViewer from '../features/system/LogViewer';
import Loading from '../Loading';
import { useAppDispatch } from './store';
import { requestSystemConfig } from './socketio/actions';
import WorkPanel from '../features/options/WorkPanel';

const App = () => {
  const dispatch = useAppDispatch();
  const [isReady, setIsReady] = useState<boolean>(false);

  useEffect(() => {
    dispatch(requestSystemConfig());
    setIsReady(true);
  }, [dispatch]);

  return isReady ? (
    <div className="App">
      <ProgressBar />
      <div className="app-content">
        <SiteHeader />
        <div className="app-content-workarea">
          <WorkPanel />
          <CurrentImageDisplay />
          <ImageGallery />
        </div>
      </div>
      <LogViewer />
    </div>
  ) : (
    <Loading />
  );
};

export default App;
