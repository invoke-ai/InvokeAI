import { Grid, GridItem } from '@chakra-ui/react';
import SiteHeader from './components/SiteHeader';
import ImageRoll from './features/gallery/ImageRoll';
import CurrentImage from './features/gallery/CurrentImage';
import SDProgress from './components/SDProgress';
import LogViewer from './components/LogViewer';
import { useSocketIOInitialize } from './context/socket';
import SDSettings from './features/sd/SDSettings';
import PromptInput from './features/sd/PromptInput';

const App = () => {
    // Loads all socket listeners & loads images to gallery
    useSocketIOInitialize();
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
                gridTemplateRows={'4px 40px 100px 1fr'}
                gridTemplateColumns={'2fr 8fr 1fr'}
                gap='2'
            >
                <GridItem area={'progressBar'}>
                    <SDProgress />
                </GridItem>
                <GridItem pl='2' pr='2' area={'header'}>
                    <SiteHeader />
                </GridItem>
                <GridItem pl='2' area={'menu'} overflowY='scroll'>
                    <SDSettings />
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
