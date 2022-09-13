import { useAppDispatch, useAppSelector } from '../../app/hooks';
import {
    cancelProcessing,
    generateImage,
    runESRGAN,
    runGFPGAN,
} from '../../app/socketio';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import useCheckParameters from '../system/useCheckParameters';

const ProcessButtons = () => {
    const areParametersValid = useCheckParameters();

    const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
        useAppSelector((state: RootState) => state.system);
    const { currentImage } = useAppSelector(
        (state: RootState) => state.gallery
    );
    const dispatch = useAppDispatch();

    return isProcessing ? (
        <SDButton
            label='Cancel'
            colorScheme='red'
            isDisabled={!isConnected || !isProcessing}
            onClick={() => dispatch(cancelProcessing())}
        />
    ) : (
        <>
            <SDButton
                label='Gen'
                type='submit'
                colorScheme='green'
                isDisabled={!areParametersValid}
                onClick={() => dispatch(generateImage())}
            />
            <SDButton
                label='Upscale'
                type='submit'
                colorScheme='green'
                isDisabled={currentImage === undefined}
                onClick={() =>
                    currentImage && dispatch(runESRGAN(currentImage))
                }
            />
            <SDButton
                label='Face'
                type='submit'
                colorScheme='green'
                isDisabled={currentImage === undefined}
                onClick={() =>
                    currentImage && dispatch(runGFPGAN(currentImage))
                }
            />
        </>
    );
};

export default ProcessButtons;
