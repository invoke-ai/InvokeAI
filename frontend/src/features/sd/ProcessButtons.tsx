import { Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { cancelProcessing, generateImage } from '../../app/socketio';
import { RootState } from '../../app/store';
import SDButton from '../../components/SDButton';
import useCheckParameters from '../system/useCheckParameters';
import { resetSDState } from './sdSlice';

const ProcessButtons = () => {
    const areParametersValid = useCheckParameters();

    const { isProcessing, isConnected } = useAppSelector(
        (state: RootState) => state.system
    );
    const dispatch = useAppDispatch();

    return (
        <Flex gap={2}>
            <SDButton
                label='Generate Image'
                type='submit'
                colorScheme='green'
                flexGrow={1}
                isDisabled={!areParametersValid}
                onClick={() => dispatch(generateImage())}
            />
            <SDButton
                label='Cancel'
                colorScheme='red'
                flexGrow={1}
                isDisabled={!isConnected || !isProcessing}
                onClick={() => dispatch(cancelProcessing())}
            />
            <SDButton
                label='Reset'
                colorScheme='blue'
                flexGrow={1}
                onClick={() => dispatch(resetSDState())}
            />
        </Flex>
    );
};

export default ProcessButtons;
