import { Grid, GridItem } from '@chakra-ui/react';
import { useSocketIOListeners } from './app/socket';
import ImageRoll from './features/gallery/ImageRoll';
import CurrentImage from './features/gallery/CurrentImage';
import LogViewer from './features/system/LogViewer';
import Settings from './features/sd/Settings';
import PromptInput from './features/sd/PromptInput';
import InitImage from './features/sd/InitImage';
import ProgressBar from './features/header/ProgressBar';
import SiteHeader from './features/header/SiteHeader';
import Variant from './features/sd/Variant';

const App = () => {
    useSocketIOListeners();
    return (
        <>
            <Grid
                width='100vw'
                height='100vh'
                templateAreas={`
                    "progressBar progressBar progressBar"
                    "header header header"
                    "menu prompt imageRoll"
                    "menu variant imageRoll"
                    "menu currentImage imageRoll"`}
                gridTemplateRows={'4px 40px 100px 32px auto'}
                gridTemplateColumns={'300px auto 388px'}
                gap='2'
            >
                <GridItem area={'progressBar'}>
                    <ProgressBar />
                </GridItem>
                <GridItem pl='2' pr='2' area={'header'}>
                    <SiteHeader />
                </GridItem>
                <GridItem pl='2' area={'menu'} overflowY='scroll'>
                    <Settings />
                    <InitImage />
                </GridItem>
                <GridItem area={'prompt'}>
                    <PromptInput />
                </GridItem>
                <GridItem area={'variant'}>
                    <Variant />
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
