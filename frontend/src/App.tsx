import { Grid, GridItem } from '@chakra-ui/react';
import SiteHeader from './components/SiteHeader';
import ImageRoll from './features/gallery/ImageRoll';
import CurrentImage from './features/gallery/CurrentImage';
import SDProgress from './components/SDProgress';
import LogViewer from './components/LogViewer';
import { useSocketIOListeners } from './context/socket';
import SDSettings from './features/sd/SDSettings';
import PromptInput from './features/sd/PromptInput';
import SDFileUpload from './components/SDFileUpload';

const App = () => {
    // Loads all socket listeners
    useSocketIOListeners();
    return (
        <>
            <Grid
                width='100vw'
                height='100vh'
                templateAreas={`
                    "progressBar progressBar progressBar"
                    "header header header"
                    "menu prompt prompt"
                    "menu currentImage imageRoll"`}
                gridTemplateRows={'4px 40px 100px auto'}
                gridTemplateColumns={'300px 4fr 150px'}
                gap='2'
            >
                <GridItem area={'progressBar'}>
                    <SDProgress />
                </GridItem>
                <GridItem pl='2' pr='2' area={'header'}>
                    <SiteHeader />
                </GridItem>
                <GridItem pl='2' area={'menu'} overflowY='scroll'>
                    <SDSettings height={475} />
                    <SDFileUpload />
                </GridItem>

                <GridItem pr='2' area={'prompt'}>
                    <PromptInput />
                </GridItem>
                <GridItem area={'currentImage'}>
                    <CurrentImage />
                </GridItem>
                <GridItem pr='2' area={'imageRoll'} overflowY='scroll'>
                    <ImageRoll />
                </GridItem>
            </Grid>
            <LogViewer />
        </>
    );
};

export default App;
