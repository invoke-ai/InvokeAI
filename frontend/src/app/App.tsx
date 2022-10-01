import { useEffect, useState } from 'react';
import ProgressBar from '../features/system/ProgressBar';
import SiteHeader from '../features/system/SiteHeader';
import LogViewer from '../features/system/LogViewer';
import Loading from '../Loading';
import { useAppDispatch } from './store';
import { requestSystemConfig } from './socketio/actions';
import { keepGUIAlive } from './utils';
import { Tab, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import TextToImage from '../features/options/TextToImage/TextToImage';
import { WorkInProgress } from '../common/components/WorkInProgress';

keepGUIAlive();

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
        <Tabs className="app-tabs">
          <div className="app-tabs-list">
            <Tab>Text To Image</Tab>
            <Tab>Image To Image</Tab>
            <Tab>Inpainting</Tab>
            <Tab>Outpainting</Tab>
            <Tab>Nodes</Tab>
            <Tab>Post Processing</Tab>
          </div>
          <TabPanels>
            <TabPanel>
              <TextToImage />
            </TabPanel>
            <TabPanel>
              <WorkInProgress />
            </TabPanel>
            <TabPanel>
              <WorkInProgress />
            </TabPanel>
            <TabPanel>
              <WorkInProgress />
            </TabPanel>
            <TabPanel>
              <WorkInProgress />
            </TabPanel>
            <TabPanel>
              <WorkInProgress />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </div>
      <LogViewer />
    </div>
  ) : (
    <Loading />
  );
};

export default App;
